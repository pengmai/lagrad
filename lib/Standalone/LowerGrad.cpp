#include "Standalone/Passes.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "Standalone/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "queue"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
class GradOpLowering : public ConversionPattern {
public:
  explicit GradOpLowering(MLIRContext *context)
      : ConversionPattern(standalone::GradOp::getOperationName(), /*benefit=*/1,
                          context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = getFunctionDeclaration(op, operands, rewriter);
    if (!funcOp) {
      return failure();
    }

    auto funcVal = rewriter.create<mlir::ConstantOp>(
        op->getLoc(), funcOp.getType(),
        SymbolRefAttr::get(funcOp.getContext(), funcOp.getName()));
    op->replaceAllUsesWith(llvm::makeArrayRef(funcVal.getResult()));
    rewriter.eraseOp(op);
    // llvm::outs() << "\n\nFuncOp:\n" << funcOp << "\n";
    return success();
  }

private:
  static bool verifyAttributes(Operation *op, unsigned int num_inputs) {
    ArrayAttr ofAttr = op->getAttr("of").dyn_cast_or_null<ArrayAttr>();
    if (ofAttr) {
      for (const auto attr : ofAttr) {
        auto intAttr = attr.dyn_cast_or_null<IntegerAttr>();
        if (!intAttr) {
          return false;
        }

        auto attrValue = intAttr.getValue().getSExtValue();
        if (attrValue < 0) {
          op->emitError("'of' index cannot be negative");
          return false;
        } else if (static_cast<size_t>(attrValue) >= num_inputs) {
          op->emitError(
              "'of' index cannot be greater than number of function inputs");
          return false;
        }
      }
    }
    return true;
  }

  static FuncOp getFunctionDeclaration(Operation *gradOp,
                                       ArrayRef<Value> operands,
                                       ConversionPatternRewriter &rewriter) {
    Value arg0 = operands[0];
    auto arg0Type = arg0.getType().dyn_cast<FunctionType>();
    if (!arg0Type) {
      gradOp->emitError("Argument to `standalone.grad` was not a function");
      return nullptr;
    }

    if (!verifyAttributes(gradOp, arg0Type.getNumInputs())) {
      return nullptr;
    }
    // auto returnShapedType =
    //     arg0Type.getResult(0).dyn_cast_or_null<ShapedType>();
    // if (!(arg0Type.getResult(0).isa<FloatType>() ||
    //       (returnShapedType && returnShapedType.hasRank() &&
    //        returnShapedType.getRank() == 0))) {
    //   gradOp->emitError("Argument to `standalone.grad` must return a float "
    //                     "type or a rank-0 shaped type");
    //   return nullptr;
    // }

    // Assume the arg was defined by a constant op.
    auto definingOp = arg0.getDefiningOp();
    if (!definingOp) {
      gradOp->emitError("Argument had no defining op");
      return nullptr;
    }

    if (!definingOp->hasAttr("value")) {
      definingOp->emitError("Expected constant op to have 'value' attribute "
                            "(trying to look up function name)");
      return nullptr;
    }

    auto attr = definingOp->getAttrOfType<FlatSymbolRefAttr>("value");
    auto moduleOp = gradOp->getParentOfType<ModuleOp>();
    auto originalFuncOp = moduleOp.lookupSymbol<FuncOp>(attr);
    auto gradientsOf = gradOp->getAttr("of").dyn_cast_or_null<ArrayAttr>();

    std::string adjointFuncName("__grad_");
    adjointFuncName += originalFuncOp.getName();

    FuncOp funcOp =
        copyFunctionDeclaration(originalFuncOp, adjointFuncName, rewriter);
    return differentiateFunction(funcOp, gradientsOf, rewriter,
                                 /*topLevel=*/true);
  }
};
} // end anonymous namespace

namespace {
struct GradTarget : public ConversionTarget {
  GradTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<mlir::StandardOpsDialect>();
    addLegalDialect<mlir::arith::ArithmeticDialect>();
    addLegalDialect<mlir::math::MathDialect>();
    addLegalDialect<mlir::memref::MemRefDialect>();
    addLegalDialect<tensor::TensorDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalDialect<linalg::LinalgDialect>();
    addLegalOp<FuncOp>();
  }
};
} // end anonymous namespace

namespace {
struct GradConversionPass
    : public PassWrapper<GradConversionPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect>();
  }
  StringRef getArgument() const override { return "take-grads"; }
  StringRef getDescription() const override {
    return "Run the autodiff procedure for standalone.grad";
  }
  void runOnOperation() final;
};
} // end anonymous namespace

void GradConversionPass::runOnOperation() {
  GradTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  OwningRewritePatternList patterns(&getContext());
  patterns.insert<GradOpLowering>(&getContext());

  auto mod = getOperation();
  if (failed(applyFullConversion(mod, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::Standalone::createGradPass() {
  return std::make_unique<GradConversionPass>();
}