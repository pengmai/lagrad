#include "Standalone/Passes.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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
    auto funcDeclResult = getFunctionDeclaration(op, operands, rewriter);
    if (funcDeclResult.value == LogicalResult::Failure) {
      return funcDeclResult;
    }
    op->replaceAllUsesWith(operands);
    rewriter.eraseOp(op);
    return success();
  }

private:
  static LogicalResult
  getFunctionDeclaration(Operation *gradOp, ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) {
    Value arg0 = operands[0];
    auto arg0Type = arg0.getType().dyn_cast<FunctionType>();
    if (!arg0Type) {
      gradOp->emitError("Argument to `standalone.grad` was not a function");
      return failure();
    }
    if (!arg0Type.getResult(0).isa<FloatType>()) {
      gradOp->emitError(
          "Argument to `standalone.grad` must return a float type");
      return failure();
    }

    // Assume the arg was defined by a constant op.
    auto definingOp = arg0.getDefiningOp();
    if (!definingOp) {
      gradOp->emitError("Argument had no defining op");
      return failure();
    }

    if (!definingOp->hasAttr("value")) {
      definingOp->emitError("Expected constant op to have 'value' attribute "
                            "(trying to look up function name)");
      return failure();
    }

    auto attr = definingOp->getAttrOfType<FlatSymbolRefAttr>("value");
    auto moduleOp = gradOp->getParentOfType<ModuleOp>();
    auto funcOp = moduleOp.lookupSymbol<FuncOp>(attr);

    auto region = funcOp.getCallableRegion();
    if (!region) {
      definingOp->emitError("Function region cannot be null");
      return failure();
    }

    std::vector<Operation *> ops;
    region->walk([&](Operation *op) { ops.push_back(op); });

    // env maps values to their gradient signals. x -> x_bar
    llvm::DenseMap<Value, Value> env;
    auto f32Ty = FloatType::getF32(region->getContext());

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    for (auto it = ops.rbegin(); it != ops.rend(); it++) {
      Operation *op = *it;
      // llvm::outs() << "op: " << *op << "\n"
      //              << "results: " << op->getNumResults() << "\n";
      if (op->isKnownTerminator()) {
        // This is the exit point
        rewriter.setInsertionPoint(op);
        Value operand = op->getOperand(0);
        // Initialize the gradient signal to 1.0
        env[operand] = rewriter.create<mlir::ConstantOp>(
            op->getLoc(), FloatAttr::get(f32Ty, 1.0));
        rewriter.eraseOp(op);
      } else {
        int op_index = 0;
        for (Value operand : op->getOperands()) {
          if (!env[operand]) {
            // TODO: This might not be a scalar
            env[operand] = rewriter.create<mlir::ConstantOp>(
                op->getLoc(), FloatAttr::get(operand.getType(), 0.0));
          }

          // Compute the pullback (VJP)
          Value vjp_value = env[op->getResult(0)];
          if (op->getName().getStringRef() == "std.mulf") {
            vjp_value = rewriter.create<mlir::MulFOp>(
                op->getLoc(), vjp_value, op->getOperand(1 - op_index));
          } else if (op->getName().getStringRef() == "std.addf") {
            // This has no effect on the VJP
          }

          // Add the gradient signals.
          env[operand] = rewriter.create<mlir::AddFOp>(op->getLoc(),
                                                       env[operand], vjp_value);
          // llvm::outs() << "operand value: " << operand << "\n";
          op_index++;
        }
        // llvm::outs() << "\n";
      }
    }

    rewriter.create<mlir::ReturnOp>(region->getLoc(),
                                    env[region->getArgument(0)]);
    return success();
  }
};
} // end anonymous namespace

namespace {
struct GradTarget : public ConversionTarget {
  GradTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<mlir::StandardOpsDialect>();
    addLegalOp<FuncOp>();
  }
};
} // end anonymous namespace

namespace {
struct GradConversionPass
    : public PassWrapper<GradConversionPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {}
  void runOnOperation() final;
};
} // end anonymous namespace

void GradConversionPass::runOnOperation() {
  GradTarget target(getContext());
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  OwningRewritePatternList patterns;
  patterns.insert<GradOpLowering>(&getContext());

  auto mod = getOperation();
  if (failed(applyFullConversion(mod, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::Standalone::createGradPass() {
  return std::make_unique<GradConversionPass>();
}