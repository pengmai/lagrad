/**
 * A custom pass meant to fill gaps in bufferizing tensor ops.
 * Motivated by a lack of bufferization for the tensor.insert op.
 */
#include "mlir/Transforms/Bufferize.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
class BufferizeInsertOp : public OpConversionPattern<tensor::InsertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // This first implementation updates the tensor in place rather than
    // returning a copy per the spec. Watch out for bugs this may cause.
    rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.scalar(),
                                     adaptor.dest(), adaptor.indices());
    auto loaded =
        rewriter.create<memref::TensorLoadOp>(op.getLoc(), adaptor.dest());
    op.replaceAllUsesWith(loaded.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
class BufferizeExtractSliceOp
    : public OpConversionPattern<tensor::ExtractSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "Running transformation\n";
    return failure();
  }
};
} // namespace

namespace {
struct StandaloneBufferizePass
    : public PassWrapper<StandaloneBufferizePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
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
    target.addIllegalOp<tensor::InsertOp>();
    target.addLegalDialect<scf::SCFDialect>();
    populateBufferizeMaterializationLegality(target);

    patterns.add<BufferizeInsertOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeExtractSliceOp>(typeConverter, patterns.getContext());
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
