/**
 * A pass to convert LLVM heap allocations to static sizes for compatibility
 * with Enzyme.
 */
#include "LAGrad/Passes.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
class ConvertStaticAlloca : public OpRewritePattern<LLVM::AllocaOp> {
public:
  ConvertStaticAlloca(MLIRContext *context)
      : OpRewritePattern<LLVM::AllocaOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(LLVM::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    auto ptrToIntOp =
        dyn_cast_or_null<LLVM::PtrToIntOp>(op.getOperand().getDefiningOp());
    if (!ptrToIntOp) {
      return failure();
    }

    auto gepOp =
        dyn_cast_or_null<LLVM::GEPOp>(ptrToIntOp.getOperand().getDefiningOp());
    if (!gepOp) {
      return failure();
    }

    auto elementType = ptrToIntOp.getArg()
                           .getType()
                           .dyn_cast<LLVM::LLVMPointerType>()
                           .getElementType();

    size_t allocSize = elementType.getIntOrFloatBitWidth() / 8;
    for (auto indexVal : gepOp.getIndices()) {
      auto indexConstOp =
          dyn_cast_or_null<LLVM::ConstantOp>(indexVal.getDefiningOp());
      assert(indexConstOp &&
             "Expected index value to be defined by a constant op");
      allocSize *= indexConstOp.getValue()
                       .dyn_cast<IntegerAttr>()
                       .getValue()
                       .getSExtValue();
    }

    auto staticSize = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(allocSize));
    rewriter.updateRootInPlace(op, [&]() { op.setOperand(staticSize); });
    return success();
  }
};

class ConvertStaticMalloc : public RewritePattern {
public:
  ConvertStaticMalloc(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto callOp = dyn_cast_or_null<LLVM::CallOp>(op);
    if (!callOp || callOp.getCalleeAttr().getValue() != "malloc") {
      return failure();
    }

    // This depends on the specific lowering generated from memref.alloc() ops.
    assert(callOp.getNumOperands() == 1 &&
           "Expected malloc() call to have one operand");
    auto ptrToIntOp = dyn_cast_or_null<LLVM::PtrToIntOp>(
        callOp.getOperand(0).getDefiningOp());
    if (!ptrToIntOp) {
      return failure();
    }

    auto gepOp =
        dyn_cast_or_null<LLVM::GEPOp>(ptrToIntOp.getOperand().getDefiningOp());
    if (!gepOp) {
      return failure();
    }

    auto elementType = ptrToIntOp.getArg()
                           .getType()
                           .dyn_cast<LLVM::LLVMPointerType>()
                           .getElementType();

    size_t allocSize = elementType.getIntOrFloatBitWidth() / 8;
    if (!llvm::all_of(gepOp.getIndices(), [](Value indexVal) {
          return isa<LLVM::ConstantOp>(indexVal.getDefiningOp());
        })) {
      // Enzyme appears to handle dynamic shapes okay.
      return failure();
    }

    for (auto indexVal : gepOp.getIndices()) {
      auto indexConstOp =
          dyn_cast_or_null<LLVM::ConstantOp>(indexVal.getDefiningOp());
      assert(indexConstOp &&
             "Expected index value to be defined by a constant op");
      allocSize *= indexConstOp.getValue()
                       .dyn_cast<IntegerAttr>()
                       .getValue()
                       .getSExtValue();
    }

    auto staticSize = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(allocSize));
    rewriter.updateRootInPlace(callOp,
                               [&]() { callOp.setOperand(0, staticSize); });
    return success();
  }
};
} // namespace

namespace {
struct StaticAllocsPass
    : public PassWrapper<StaticAllocsPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  StringRef getArgument() const override { return "convert-static-allocs"; }
  StringRef getDescription() const override {
    return "Convert malloc calls with statically known shapes to use constant "
           "values";
  }
  void runOnOperation() final {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertStaticMalloc>(patterns.getContext());
    patterns.add<ConvertStaticAlloca>(patterns.getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::lagrad::createStaticAllocsPass() {
  return std::make_unique<StaticAllocsPass>();
}
