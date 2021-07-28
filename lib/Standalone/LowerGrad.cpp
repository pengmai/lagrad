#include "Standalone/Passes.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
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
    auto funcOp = getFunctionDeclaration(op, operands, rewriter);
    if (!funcOp) {
      return failure();
    }

    auto funcVal = rewriter.create<mlir::ConstantOp>(
        op->getLoc(), funcOp.getType(),
        SymbolRefAttr::get(funcOp.getName(), funcOp.getContext()));
    op->replaceAllUsesWith(llvm::makeArrayRef(funcVal.getResult()));
    rewriter.eraseOp(op);
    return success();
  }

private:
  static FuncOp getFunctionDeclaration(Operation *gradOp,
                                       ArrayRef<Value> operands,
                                       ConversionPatternRewriter &rewriter) {
    Value arg0 = operands[0];
    auto arg0Type = arg0.getType().dyn_cast<FunctionType>();
    if (!arg0Type) {
      gradOp->emitError("Argument to `standalone.grad` was not a function");
      return nullptr;
    }
    // if (!arg0Type.getResult(0).isa<FloatType>()) {
    //   gradOp->emitError(
    //       "Argument to `standalone.grad` must return a float type");
    //   return failure();
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

    FuncOp funcOp = copyFunctionDeclaration(originalFuncOp, rewriter);

    Region *region = funcOp.getCallableRegion();
    if (!region) {
      definingOp->emitError("Function region cannot be null");
      return nullptr;
    }

    std::vector<Operation *> ops;
    for (auto &op : region->getOps()) {
      ops.push_back(&op);
    }

    // env maps values to their gradient signals. x -> x_bar
    llvm::DenseMap<Value, Value> env;

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    for (auto it = ops.rbegin(); it != ops.rend(); it++) {
      Operation *op = *it;
      if (op->getName().getStringRef() == "std.return") {
        // if (op->isKnownTerminator()) {
        // This is the exit point
        rewriter.setInsertionPoint(op);
        Value operand = op->getOperand(0);
        // Initialize the gradient signal to 1.0
        env[operand] = onesLike(op->getLoc(), operand, rewriter);
        rewriter.eraseOp(op);
      } else {
        int op_index = 0;
        for (Value operand : op->getOperands()) {
          if (!env[operand]) {
            env[operand] = getZero(op->getLoc(), operand, rewriter);
          }

          // Compute the pullback (VJP).
          // TODO: Gotta be a better way to structure/abstract this.
          if (op->getNumResults() == 0) {
            op->emitError("op had zero results");
            return nullptr;
          }
          Value vjp_value = env[op->getResult(0)];
          // TODO: To what extent do I need this? I put it in as hack to avoid
          // changing the function type.
          if (!vjp_value) {
            Value result = op->getResult(0);
            vjp_value = onesLike(result.getLoc(), result, rewriter);
            env[result] = vjp_value;
            // llvm::outs() << "Initializing value (not yet initialized): "
            //              << result << "\n";
          }
          auto opName = op->getName().getStringRef();
          if (opName == "std.mulf") {
            vjp_value = rewriter.create<mlir::MulFOp>(
                op->getLoc(), vjp_value, op->getOperand(1 - op_index));
          } else if (opName == "std.addf") {
            // This has no effect on the VJP
          } else if (opName == "std.subf") {
            if (op_index == 1) {
              vjp_value =
                  rewriter.create<mlir::NegFOp>(op->getLoc(), vjp_value);
            }
          } else if (opName == "std.divf") {
            if (op_index == 0) {
              vjp_value = rewriter.create<mlir::DivFOp>(op->getLoc(), vjp_value,
                                                        op->getOperand(1));
            } else {
              assert(op_index == 1 && "std.divf op had more than 2 args");
              vjp_value = rewriter.create<mlir::MulFOp>(op->getLoc(), vjp_value,
                                                        op->getOperand(0));
              vjp_value =
                  rewriter.create<mlir::NegFOp>(op->getLoc(), vjp_value);
              Value denom =
                  rewriter.create<mlir::MulFOp>(op->getLoc(), operand, operand);
              vjp_value =
                  rewriter.create<mlir::DivFOp>(op->getLoc(), vjp_value, denom);
            }
          } else if (opName == "std.negf") {
            vjp_value = rewriter.create<mlir::NegFOp>(op->getLoc(), vjp_value);
          } else if (opName == "linalg.dot") {
            if (op_index > 1)
              continue;

            Value broadcast_value =
                rewriter.create<tensor::ExtractOp>(operand.getLoc(), vjp_value);

            assert(operand.getType().isa<ShapedType>() &&
                   "dot input arg was not a shaped type");
            auto opType = operand.getType().dyn_cast<ShapedType>();
            // The gradient is the other argument.
            vjp_value =
                broadcast(operand.getLoc(), opType, broadcast_value, rewriter);
            vjp_value = rewriter.create<mlir::MulFOp>(
                operand.getLoc(), op->getOperand(1 - op_index), vjp_value);
          } else if (opName == "tensor.extract") {
            // TODO: This only supports 0d tensors
            op->emitError("differentiating tensor.extract not yet supported");
            // Value index = rewriter.create<mlir::ConstantOp>(
            //     operand.getLoc(),
            //     IntegerAttr::get(IndexType::get(operand.getContext()), 0));
            // auto genOp = rewriter.create<tensor::GenerateOp>(
            //     operand.getLoc(), env[operand].getType(), index);
            // vjp_value = genOp.getResult();
          } else {
            llvm::outs() << "Unrecognized op: " << op->getName().getStringRef()
                         << "\n";
          }

          // Add the gradient signals.
          env[operand] = rewriter.create<mlir::AddFOp>(op->getLoc(),
                                                       env[operand], vjp_value);
          op_index++;
        }
      }
    }

    auto fntyp = funcOp.getType();
    funcOp.setType(FunctionType::get(funcOp.getContext(), fntyp.getInputs(),
                                     {region->getArgument(0).getType()}));
    rewriter.create<mlir::ReturnOp>(region->getLoc(),
                                    env[region->getArgument(0)]);
    return funcOp;
  }

  static FuncOp copyFunctionDeclaration(FuncOp funcOp,
                                        ConversionPatternRewriter &rewriter) {
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointAfter(funcOp);
    auto newOp = static_cast<FuncOp>(rewriter.clone(*funcOp));

    std::string gradFuncName = "__grad_";
    gradFuncName += funcOp.getName();
    newOp.setName(gradFuncName);
    return newOp;
  }

  static mlir::Value getZero(Location loc, mlir::Value operand,
                             ConversionPatternRewriter &rewriter) {
    if (operand.getType().isa<FloatType>()) {
      return rewriter.create<mlir::ConstantOp>(
          loc, FloatAttr::get(operand.getType(), 0.0));
    }
    if (operand.getType().isa<ShapedType>()) {
      auto shapedType = operand.getType().dyn_cast<ShapedType>();
      // Will automatically be broadcasted to the right shape.
      auto denseAttr = DenseFPElementsAttr::get(shapedType, {0.0f});
      return rewriter.create<mlir::ConstantOp>(loc, denseAttr);
    }
    llvm_unreachable("not yet implemented");
    return nullptr;
  }

  static mlir::Value onesLike(Location loc, mlir::Value operand,
                              ConversionPatternRewriter &rewriter) {
    if (operand.getType().isa<FloatType>()) {
      return rewriter.create<mlir::ConstantOp>(
          loc, FloatAttr::get(operand.getType(), 1.0));
    }
    if (operand.getType().isa<ShapedType>()) {
      auto shapedType = operand.getType().dyn_cast<ShapedType>();
      auto denseAttr = DenseFPElementsAttr::get(shapedType, {1.0f});
      return rewriter.create<mlir::ConstantOp>(loc, denseAttr);
    }
    llvm_unreachable("ones for type not yet implemented");
    return nullptr;
  }

  static mlir::Value broadcast(Location loc, Type type,
                               mlir::Value broadcast_value,
                               ConversionPatternRewriter &rewriter) {
    llvm::SmallVector<Value> empty;
    auto generateOp =
        rewriter.create<mlir::tensor::GenerateOp>(loc, type, empty);

    OpBuilder::InsertionGuard guard(rewriter);

    rewriter.createBlock(&generateOp.getBodyRegion(), {},
                         TypeRange({IndexType::get(loc.getContext())}));
    rewriter.create<mlir::tensor::YieldOp>(loc, broadcast_value);
    return generateOp;
  }
};
} // end anonymous namespace

namespace {
struct GradTarget : public ConversionTarget {
  GradTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<mlir::StandardOpsDialect>();
    addLegalDialect<tensor::TensorDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
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
  target.addLegalOp<linalg::DotOp, linalg::YieldOp, linalg::FillOp>();

  OwningRewritePatternList patterns;
  patterns.insert<GradOpLowering>(&getContext());

  auto mod = getOperation();
  if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::Standalone::createGradPass() {
  return std::make_unique<GradConversionPass>();
}