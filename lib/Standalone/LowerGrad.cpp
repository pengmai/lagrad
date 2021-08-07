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
        // This is the exit point
        rewriter.setInsertionPoint(op);
        Value operand = op->getOperand(0);
        // Initialize the gradient signal to 1.0
        env[operand] = onesLike(op->getLoc(), operand, rewriter);
        rewriter.eraseOp(op);
      } else {
        int op_index = 0;
        for (Value operand : op->getOperands()) {
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
          } else if (opName == "std.exp") {
            assert(op->getNumResults() == 1 &&
                   "math.exp op did not have exactly 1 result");
            vjp_value = rewriter.create<mlir::MulFOp>(op->getLoc(), vjp_value,
                                                      op->getResult(0));
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
          } else if (opName == "linalg.matvec") {
            if (op_index > 1)
              continue;
            if (op_index == 0) {
              // Broadcast the gradient signal
              assert(operand.getType().isa<RankedTensorType>() &&
                     "matvec input was not a ranked tensor type");
              SmallVector<AffineMap, 3> indexingMaps(
                  op->getNumOperands(), rewriter.getMultiDimIdentityMap(2));
              indexingMaps[0] = indexingMaps[0].getSubMap({0});
              indexingMaps[1] = indexingMaps[1].getSubMap({1});
              auto opType = operand.getType().dyn_cast<RankedTensorType>();
              SmallVector<StringRef, 6> iteratorTypes(
                  opType.getRank(), getParallelIteratorTypeName());
              auto outerProductOp = rewriter.create<linalg::GenericOp>(
                  operand.getLoc(),
                  /*resultTensorTypes=*/opType,
                  /*inputs=*/ValueRange({vjp_value, op->getOperand(1)}),
                  /*outputs=*/ValueRange({operand}), indexingMaps,
                  iteratorTypes,
                  [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
                    Value mul_res = builder.create<mlir::MulFOp>(
                        loc, regionArgs[0], regionArgs[1]);
                    builder.create<linalg::YieldOp>(loc, mul_res);
                  });
              vjp_value = outerProductOp.getResult(0);
            } else {
              Value zero = env[operand]
                               ? env[operand]
                               : getZero(operand.getLoc(), operand, rewriter);
              // TODO: Probably a more elegant way to do this. The goal is to
              // express indexingMaps =
              //   {<(d0, d1)> -> <(d0)>,
              //    <(d0, d1)> -> <(d0, d1)>
              //    <(d0, d1)> -> <(d1)>}
              SmallVector<AffineMap, 3> indexingMaps(
                  op->getNumOperands(), rewriter.getMultiDimIdentityMap(2));
              indexingMaps[0] = indexingMaps[0].getSubMap({0});
              indexingMaps[2] = indexingMaps[2].getSubMap({1});
              SmallVector<StringRef, 6> iteratorTypes(
                  {getReductionIteratorTypeName(),
                   getParallelIteratorTypeName()});

              // TODO: This currently uses the allocated gradient space and adds
              // it inside the matmul. This may produce incorrect results due to
              // being added twice? Especially down the line with bufferization.
              auto matmulOp = rewriter.create<linalg::GenericOp>(
                  operand.getLoc(),
                  /*resultTensorTypes=*/operand.getType(),
                  /*inputs=*/ValueRange({vjp_value, op->getOperand(0)}),
                  /*outputs=*/ValueRange({zero}),
                  /*indexingMaps=*/indexingMaps,
                  /*iteratorTypes=*/iteratorTypes,
                  [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
                    auto mul_res = builder.create<mlir::MulFOp>(
                        loc, regionArgs[0], regionArgs[1]);
                    auto reduced = builder.create<mlir::AddFOp>(loc, mul_res,
                                                                regionArgs[2]);
                    builder.create<linalg::YieldOp>(loc, reduced.getResult());
                  });
              vjp_value = matmulOp.getResult(0);
            }
          } else if (opName == "linalg.vecmat") {
            if (op_index > 1) {
              continue;
            } else if (op_index == 0) {
              Value zero = env[operand]
                               ? env[operand]
                               : getZero(operand.getLoc(), operand, rewriter);
              SmallVector<AffineMap, 3> indexingMaps(
                  op->getNumOperands(), rewriter.getMultiDimIdentityMap(2));
              indexingMaps[0] = indexingMaps[0].getSubMap({1});
              indexingMaps[2] = indexingMaps[2].getSubMap({0});
              SmallVector<StringRef, 6> iteratorTypes(
                  {getParallelIteratorTypeName(),
                   getReductionIteratorTypeName()});
              auto matmulOp = rewriter.create<linalg::GenericOp>(
                  operand.getLoc(),
                  /*resultTensorTypes=*/operand.getType(),
                  /*inputs=*/ValueRange({vjp_value, op->getOperand(1)}),
                  /*outputs=*/ValueRange({zero}),
                  /*indexingMaps=*/indexingMaps,
                  /*iteratorTypes=*/iteratorTypes,
                  [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
                    auto mul_res = builder.create<mlir::MulFOp>(
                        loc, regionArgs[0], regionArgs[1]);
                    auto reduced = builder.create<mlir::AddFOp>(loc, mul_res,
                                                                regionArgs[2]);
                    builder.create<linalg::YieldOp>(loc, reduced.getResult());
                  });
              vjp_value = matmulOp.getResult(0);
            } else {
              // TODO: This is almost identical to the arg 0 case of matvec
              assert(operand.getType().isa<RankedTensorType>() &&
                     "matvec input was not a ranked tensor type");
              SmallVector<AffineMap, 3> indexingMaps(
                  op->getNumOperands(), rewriter.getMultiDimIdentityMap(2));
              indexingMaps[0] = indexingMaps[0].getSubMap({1});
              indexingMaps[1] = indexingMaps[1].getSubMap({0});
              auto opType = operand.getType().dyn_cast<RankedTensorType>();
              SmallVector<StringRef, 6> iteratorTypes(
                  opType.getRank(), getParallelIteratorTypeName());
              auto outerProductOp = rewriter.create<linalg::GenericOp>(
                  operand.getLoc(),
                  /*resultTensorTypes=*/opType,
                  /*inputs=*/ValueRange({vjp_value, op->getOperand(0)}),
                  /*outputs=*/ValueRange({operand}), indexingMaps,
                  iteratorTypes,
                  [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
                    Value mul_res = builder.create<mlir::MulFOp>(
                        loc, regionArgs[0], regionArgs[1]);
                    builder.create<linalg::YieldOp>(loc, mul_res);
                  });
              vjp_value = outerProductOp.getResult(0);
            }
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
          if (!env[operand]) {
            env[operand] = vjp_value;
          } else {
            env[operand] = rewriter.create<mlir::AddFOp>(
                op->getLoc(), env[operand], vjp_value);
          }
          op_index++;
        }
      }
    }

    auto fntyp = funcOp.getType();
    // if (!env[region->getArgument(0)]) {
    //   env[region->getArgument(0)] = getZero(region->getArgument(0).getLoc(),
    //                                         region->getArgument(0),
    //                                         rewriter);
    // }
    auto gradientsOf = gradOp->getAttr("of").dyn_cast_or_null<ArrayAttr>();
    SmallVector<Type> returnType(arg0Type.getNumInputs());
    SmallVector<Value> returnValue(arg0Type.getNumInputs());
    if (gradientsOf) {
      returnType.resize(gradientsOf.size());
      returnValue.resize(gradientsOf.size());
      for (size_t i = 0; i < gradientsOf.size(); i++) {
        auto argIndex =
            gradientsOf[i].dyn_cast<IntegerAttr>().getValue().getSExtValue();
        returnType[i] = region->getArgument(argIndex).getType();
        returnValue[i] = env[region->getArgument(argIndex)];
      }
    } else {
      returnType.resize(1);
      returnType[0] = region->getArgument(0).getType();
      returnValue.resize(1);
      returnValue[0] = env[region->getArgument(0)];
    }
    funcOp.setType(
        FunctionType::get(funcOp.getContext(), fntyp.getInputs(), returnType));
    rewriter.create<mlir::ReturnOp>(region->getLoc(), returnValue);
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
    auto shapedType = type.dyn_cast<ShapedType>();
    assert(shapedType && shapedType.hasRank() &&
           "broadcast given type that was not a ranked type");

    llvm::SmallVector<Value> empty;
    auto generateOp =
        rewriter.create<mlir::tensor::GenerateOp>(loc, type, empty);

    OpBuilder::InsertionGuard guard(rewriter);

    int64_t rank = shapedType.getRank();
    llvm::SmallVector<Type> blockArgs(rank);
    for (int i = 0; i < rank; i++) {
      blockArgs[i] = IndexType::get(loc.getContext());
    }

    auto generateBlock = rewriter.createBlock(&generateOp.getBodyRegion(), {},
                                              TypeRange(blockArgs));

    if (rank == 1) {
      rewriter.create<mlir::tensor::YieldOp>(loc, broadcast_value);
    } else if (rank == 2) {
      auto broadcast_type =
          broadcast_value.getType().dyn_cast<RankedTensorType>();
      assert(broadcast_type && "broadcast value was not a ranked tensor");
      assert(broadcast_type.getRank() == 1 &&
             "broadcast to rank 2 requires a broadcast_value with rank 1");

      Value element;
      if (shapedType.getShape()[0] == broadcast_type.getShape()[0]) {
        element = rewriter.create<tensor::ExtractOp>(
            loc, broadcast_value, generateBlock->getArgument(0));
      } else if (shapedType.getShape()[1] == broadcast_type.getShape()[0]) {
        element = rewriter.create<tensor::ExtractOp>(
            loc, broadcast_value, generateBlock->getArgument(1));
      } else {
        assert(0 && "Can't broadcast 2D types without matching dims");
      }
      rewriter.create<tensor::YieldOp>(loc, element);
    } else {
      assert(0 && "broadcast for rank >= 3 not yet implemented");
    }
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
    addLegalDialect<linalg::LinalgDialect>();
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