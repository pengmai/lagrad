/**
 * A custom pass that lowers elementwise std operations to affine loops.
 */
#include "Standalone/Passes.h"
// #include "mlir/Dialect"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

// Taken from
// https://github.com/llvm/llvm-project/commit/53a0d45db6d0f33dfbb724c99ce2560ae25473c2
static bool isElementwiseMappableOpOnRankedTensors(Operation *op) {
  if (!op->hasTrait<OpTrait::ElementwiseMappable>())
    return false;

  // TODO: The conversion pattern can be made to work for `any_of` here, but
  // it's more complex as it requires tracking which operands are scalars.
  return llvm::all_of(op->getOperandTypes(),
                      [](Type type) { return type.isa<RankedTensorType>(); });
}

namespace {
class ElementwiseToAffineLowering : public RewritePattern {
public:
  explicit ElementwiseToAffineLowering(MLIRContext *context)
      : RewritePattern(/*benefit=*/1, MatchAnyOpTypeTag()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isElementwiseMappableOpOnRankedTensors(op))
      return rewriter.notifyMatchFailure(
          op, "requires elementwise op on ranked tensors");

    auto rankedType = op->getResult(0).getType().cast<RankedTensorType>();
    auto rank = op->getResult(0).getType().cast<RankedTensorType>().getRank();
    if (rank != 1) {
      op->emitError("conversion for higher ranked tensors not yet implemented");
      return failure();
    }

    Value destination = rewriter.create<mlir::AllocOp>(
        op->getLoc(),
        MemRefType::get(rankedType.getShape(), rankedType.getElementType()));
    rewriter.create<mlir::AffineForOp>(
        op->getLoc(),
        /*start=*/0, /*stop=*/rankedType.getShape()[0], /*step=*/1, llvm::None,
        [&](OpBuilder &builder, Location loc, Value value,
            ValueRange regionArgs) {
          // Copied again from the convert-elementwise-to-linalg pass
          OperationState state(loc, op->getName());
          state.addAttributes(op->getAttrs());

          Value lhs;
          // TODO: Move this to a helper function, check that defining op exists.
          // if (op->getOperand(0).getDefiningOp()->getName().getStringRef() ==
          //     "std.tensor_load") {
          //   lhs = builder.create<mlir::AffineLoadOp>(
          //       loc, op->getOperand(0).getDefiningOp()->getOperand(0), value);
          // } else {
            lhs = builder.create<tensor::ExtractOp>(loc, op->getOperand(0),
                                                    value);
          // }

          Value rhs;
          if (op->getOperand(1).getDefiningOp()->getName().getStringRef() ==
              "std.tensor_load") {
            rhs = rewriter.create<mlir::AffineLoadOp>(
                loc, op->getOperand(1).getDefiningOp()->getOperand(0), value);
          } else {
            rhs = rewriter.create<tensor::ExtractOp>(loc, op->getOperand(1),
                                                     value);
          }
          state.addOperands({lhs, rhs});
          auto resultTypes = llvm::to_vector<6>(
              llvm::map_range(op->getResultTypes(), [](Type type) {
                return type.cast<TensorType>().getElementType();
              }));
          state.addTypes(resultTypes);
          auto *scalarOp = builder.createOperation(state);
          builder.create<mlir::AffineStoreOp>(loc, scalarOp->getResult(0),
                                              destination, value);
          builder.create<mlir::AffineYieldOp>(loc);
        });
    Value result =
        rewriter.create<mlir::TensorLoadOp>(op->getLoc(), destination);

    op->replaceAllUsesWith(llvm::makeArrayRef(result));
    rewriter.eraseOp(op);
    return success();
  }
};

struct AffineTarget : public ConversionTarget {
  AffineTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<mlir::StandardOpsDialect>();
    addLegalDialect<tensor::TensorDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalOp<FuncOp>();
  }
};

struct ElementwiseToAffineConversionPass
    : public PassWrapper<ElementwiseToAffineConversionPass,
                         OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {}
  void runOnOperation() final {
    ConversionTarget target(getContext());
    // target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
    OwningRewritePatternList patterns;
    patterns.insert<ElementwiseToAffineLowering>(&getContext());

    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !isElementwiseMappableOpOnRankedTensors(op);
    });
    auto mod = getOperation();
    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::Standalone::createElementwiseToAffinePass() {
  return std::make_unique<ElementwiseToAffineConversionPass>();
}
