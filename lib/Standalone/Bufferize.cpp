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
constexpr bool in_place = false;

namespace {
class BufferizeInsertOp : public OpConversionPattern<tensor::InsertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    memref::TensorLoadOp loaded;
    if (in_place) {
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
    if (!resultTensorType || resultRank > 2 ||
        op.getOffsetSizeAndStrideStartOperandIndex() != 1) {
      return failure();
    }

    auto slice_layout = getRankReduceSubviewLayout(resultRank, rewriter);
    auto resultType =
        MemRefType::get(resultTensorType.getShape(),
                        resultTensorType.getElementType(), slice_layout);
    auto identityResultType =
        getTypeConverter()->convertType(resultTensorType).cast<MemRefType>();

    auto subview = rewriter.create<memref::SubViewOp>(
        op.getLoc(), resultType, adaptor.source(), op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());
    if (in_place) {
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
    // Update the buffer in place. This is potentially a dangerous operation.
    auto destType = getTypeConverter()
                        ->convertType(adaptor.source().getType())
                        .dyn_cast<MemRefType>();

    auto slice_layout =
        getRankReduceSubviewLayout(op.getSourceType().getRank(), rewriter);
    auto sliceType = MemRefType::get(destType.getShape(),
                                     destType.getElementType(), slice_layout);
    auto make_copy = op->getAttr("make_copy").dyn_cast_or_null<BoolAttr>();
    // if (make_copy && make_copy.getValue()) {
    //   auto dest = rewriter.create<memref::AllocOp>(
    //       op.getLoc(), adaptor.dest().getType().dyn_cast<MemRefType>());
    //   rewriter.create<linalg::CopyOp>(op.getLoc(), adaptor.dest(), dest);
    //   auto subview = rewriter.create<memref::SubViewOp>(
    //       op.getLoc(), sliceType, dest, op.getMixedOffsets(),
    //       op.getMixedSizes(), op.getMixedStrides());

    //   rewriter.create<linalg::CopyOp>(op.getLoc(), adaptor.source(),
    //   subview); rewriter.replaceOp(op, dest.getResult());
    // } else {
    //   auto subview = rewriter.create<memref::SubViewOp>(
    //       op.getLoc(), sliceType, adaptor.dest(), op.getMixedOffsets(),
    //       op.getMixedSizes(), op.getMixedStrides());
    //   rewriter.create<linalg::CopyOp>(op.getLoc(), adaptor.source(),
    //   subview); rewriter.replaceOp(op, adaptor.dest());
    // }
    auto insert_slice_in_place = true;
    if (insert_slice_in_place) {
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
