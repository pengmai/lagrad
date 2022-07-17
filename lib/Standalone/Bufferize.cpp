/**
 * A custom pass meant to fill gaps in bufferizing tensor ops.
 * Motivated by a lack of bufferization for the tensor.insert op.
 */
#include "mlir/Transforms/Bufferize.h"
#include "Standalone/Passes.h"
#include "Standalone/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
// constexpr bool in_place = false;

namespace {
class BufferizeInsertOp : public OpConversionPattern<tensor::InsertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    memref::TensorLoadOp loaded;

    // This is very naïve.
    DominanceInfo dom;
    bool hasUseAfter = false;

    for (auto user : op.dest().getUsers()) {
      if (dom.properlyDominates(op.getResult(), user)) {
        hasUseAfter = true;
      }
    }
    // constexpr bool bufferize_insert_in_place = false;
    // if (bufferize_insert_in_place) {
    if (!hasUseAfter) {
      // This first implementation updates the tensor in place rather than
      // returning a copy per the spec. Watch out for bugs this may cause.
      rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.scalar(),
                                       adaptor.dest(), adaptor.indices());
      loaded =
          rewriter.create<memref::TensorLoadOp>(op.getLoc(), adaptor.dest());
    } else {
      auto space = rewriter.create<memref::AllocOp>(
          op.getLoc(), adaptor.dest().getType().cast<MemRefType>());
      rewriter.create<linalg::CopyOp>(op.getLoc(), adaptor.dest(), space);
      rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.scalar(), space,
                                       adaptor.indices());
      loaded = rewriter.create<memref::TensorLoadOp>(op.getLoc(), space);
    }

    op.replaceAllUsesWith(loaded.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {

/**
 * Replace tensor.extract_slice ops with a combination of memref.subview and
 * memref.cast. This operation is fragile and will produce incorrect results if
 * the result carries over function lines.
 */
class BufferizeExtractSliceOp
    : public OpConversionPattern<tensor::ExtractSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getDroppedDims().empty()) {
      // Only deal with rank reduced cases
      return failure();
    }
    auto resultTensorType =
        op.getResult().getType().dyn_cast_or_null<RankedTensorType>();
    auto resultRank = resultTensorType.getRank();
    // Curently only deal with 2D -> 1D and 3D -> 2D rank reduction cases.
    if (!resultTensorType ||
        op.getOffsetSizeAndStrideStartOperandIndex() != 1) {
      return failure();
    }

    // the getRankReduceSubviewLayout is deprecated but we want to continue
    // using it to ensure that the resulting layout is fully dynamic for
    // compatibility with our partial bufferization.
    auto slice_layout = getRankReduceSubviewLayout(resultRank, rewriter);
    auto resultType =
        MemRefType::get(resultTensorType.getShape(),
                        resultTensorType.getElementType(), slice_layout);
    auto identityResultType =
        getTypeConverter()->convertType(resultTensorType).cast<MemRefType>();

    auto subview = rewriter.create<memref::SubViewOp>(
        op.getLoc(), resultType, adaptor.source(), op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());

    DominanceInfo dom;
    bool hasWriteAfter = false;

    for (auto user : op.source().getUsers()) {
      if (dom.properlyDominates(op.getResult(), user) &&
          (dyn_cast<tensor::InsertSliceOp>(user) ||
           dyn_cast<tensor::InsertOp>(user))) {
        hasWriteAfter = true;
      }
    }
    if (!hasWriteAfter) {
      rewriter.replaceOpWithNewOp<memref::CastOp>(op, subview.getResult(),
                                                  identityResultType);
    } else {
      auto dest =
          rewriter.create<memref::AllocOp>(op.getLoc(), identityResultType);
      rewriter.create<linalg::CopyOp>(op.getLoc(), subview, dest);
      rewriter.replaceOp(op, dest.getResult());
    }
    return success();
  }
};
} // namespace

namespace {
class BufferizeInsertSliceOp
    : public OpConversionPattern<tensor::InsertSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DominanceInfo dom;
    bool hasUseAfter = false;

    for (auto user : op.dest().getUsers()) {
      if (dom.properlyDominates(op.getResult(), user)) {
        hasUseAfter = true;
      }
    }

    auto sliceType =
        memref::SubViewOp::inferRankReducedResultType(
            op.getSourceType().getRank(),
            adaptor.dest().getType().cast<MemRefType>(), op.getMixedOffsets(),
            op.getMixedSizes(), op.getMixedStrides())
            .cast<MemRefType>();
    if (!hasUseAfter) {
      auto subview = rewriter.create<memref::SubViewOp>(
          op.getLoc(), sliceType, adaptor.dest(), op.getMixedOffsets(),
          op.getMixedSizes(), op.getMixedStrides());
      rewriter.create<linalg::CopyOp>(op.getLoc(), adaptor.source(), subview);
      rewriter.replaceOp(op, adaptor.dest());
    } else {
      auto dest = rewriter.create<memref::AllocOp>(
          op.getLoc(), adaptor.dest().getType().dyn_cast<MemRefType>());
      rewriter.create<linalg::CopyOp>(op.getLoc(), adaptor.dest(), dest);
      auto subview = rewriter.create<memref::SubViewOp>(
          op.getLoc(), sliceType, dest, op.getMixedOffsets(),
          op.getMixedSizes(), op.getMixedStrides());

      rewriter.create<linalg::CopyOp>(op.getLoc(), adaptor.source(), subview);
      rewriter.replaceOp(op, dest.getResult());
    }
    return success();
  }
};
} // namespace

namespace {
struct StandaloneBufferizePass
    : public PassWrapper<StandaloneBufferizePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, linalg::LinalgDialect>();
  }
  StringRef getArgument() const override { return "standalone-bufferize"; }
  StringRef getDescription() const override {
    return "Bufferize tensor ops required by the standalone dialect that "
           "aren't bufferized elsewhere.";
  }
  void runOnOperation() final {
    auto *context = &getContext();
    BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    target.addLegalDialect<memref::MemRefDialect>();
    target.addDynamicallyLegalDialect<arith::ArithmeticDialect,
                                      StandardOpsDialect>(
        [&](Operation *op) { return typeConverter.isLegal(op); });
    target.addLegalOp<CallOp>();
    target.addLegalOp<ReturnOp>();
    target.addLegalOp<linalg::CopyOp>();
    target.addLegalOp<linalg::YieldOp>();
    target.addIllegalOp<tensor::InsertOp>();
    target.addLegalDialect<scf::SCFDialect>();
    populateBufferizeMaterializationLegality(target);

    patterns.add<BufferizeInsertOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeExtractSliceOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeInsertSliceOp>(typeConverter, patterns.getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {

      signalPassFailure();
    }
  };
};
} // namespace

std::unique_ptr<Pass> mlir::Standalone::createBufferizePass() {
  return std::make_unique<StandaloneBufferizePass>();
}
