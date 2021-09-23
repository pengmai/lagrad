#include "Standalone/Passes.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "queue"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
mlir::Value onesLike(Location loc, mlir::Value operand, OpBuilder &builder) {
  if (operand.getType().isa<FloatType>()) {
    return builder.create<mlir::ConstantOp>(
        loc, FloatAttr::get(operand.getType(), 1.0));
  }
  if (operand.getType().isa<ShapedType>()) {
    auto shapedType = operand.getType().dyn_cast<ShapedType>();
    auto denseAttr = shapedType.getElementTypeBitWidth() == 32
                         ? DenseFPElementsAttr::get(shapedType, {1.0f})
                         : DenseFPElementsAttr::get(shapedType, {1.0});
    return builder.create<mlir::ConstantOp>(loc, denseAttr);
  }
  llvm::outs() << "ones for type " << operand.getType() << " not implemented\n";
  llvm_unreachable("");
  return nullptr;
}

SmallVector<mlir::Operation *>
cloneLinalgGenericBasicBlock(linalg::GenericOp genericOp, OpBuilder &builder,
                             ValueRange regionArgs,
                             SmallVector<Value> genericOperands) {
  SmallVector<mlir::Operation *> genericRegionOps;
  DenseMap<Value, Value> old_to_new;
  for (size_t i = 0; i < genericOperands.size(); i++) {
    // The last generic operand is shifted by one. It corresponds to the output
    // in the primal, but the gradient signal is inserted at the end of the
    // adjoint, hence the shift.
    if (i == genericOperands.size() - 1) {
      old_to_new[genericOperands[i]] = regionArgs[genericOperands.size()];
    } else {
      old_to_new[genericOperands[i]] = regionArgs[i];
    }
  }

  for (auto &op : genericOp.getOps()) {
    auto clonedOp = builder.clone(op);
    for (size_t i = 0; i < clonedOp->getNumOperands(); i++) {
      assert(op.getNumResults() < 2 &&
             "linalg.generic bb op with more than two results "
             "not yet "
             "supported");
      // We assume that region arguments and intermediate values will populate
      // this map. If an entry is missing, it should have been defined outside
      // the linalg.generic body.
      if (old_to_new[clonedOp->getOperand(i)]) {
        clonedOp->setOperand(i, old_to_new[clonedOp->getOperand(i)]);
      }
      if (op.getNumResults() == 1) {
        old_to_new[op.getResult(0)] = clonedOp->getResult(0);
      }
    }
    genericRegionOps.push_back(clonedOp);
  }

  return genericRegionOps;
}

void populateVJP(Operation *op, llvm::DenseMap<Value, Value> &env,
                 OpBuilder &rewriter) {
  auto opName = op->getName().getStringRef();
  if (opName == "std.sitofp") {
    // The input is an integer so can't have a gradient signal.
    return;
  }

  size_t op_index = 0;
  for (Value operand : op->getOperands()) {
    // Compute the pullback (VJP).
    // TODO: Gotta be a better way to structure/abstract this. It's
    // essentially a huge switch statement on the operator name.
    if (op->getNumResults() == 0) {
      // op->emitError("op had zero results");
      llvm_unreachable("op had zero results");
      return;
    }
    Value vjp_value = env[op->getResult(0)];
    // TODO: To what extent do I need this? I put it in as hack to avoid
    // changing the function type.
    if (!vjp_value) {
      Value result = op->getResult(0);
      vjp_value = onesLike(result.getLoc(), result, rewriter);
      env[result] = vjp_value;
      llvm::outs() << "Initializing value (not yet initialized): " << result
                   << "\n";
    }

    if (opName == "std.mulf") {
      vjp_value = rewriter.create<mlir::MulFOp>(op->getLoc(), vjp_value,
                                                op->getOperand(1 - op_index));
    } else if (opName == "std.addf") {
      // This has no effect on the VJP
    } else if (opName == "std.subf") {
      if (op_index == 1) {
        vjp_value = rewriter.create<mlir::NegFOp>(op->getLoc(), vjp_value);
      }
    } else if (opName == "std.select") {
      // auto selectOp = dyn_cast<mlir::SelectOp>(op);
      llvm_unreachable("std.select not yet implemented");
    } else if (opName == "math.exp") {
      assert(op->getNumResults() == 1 &&
             "math.exp op did not have exactly 1 result");
      vjp_value = rewriter.create<mlir::MulFOp>(op->getLoc(), vjp_value,
                                                op->getResult(0));
    } else {
      llvm::outs() << "(linalg.generic) unrecognized op: " << opName << "\n";
    }

    // Add the gradient signals.
    if (!env[operand]) {
      env[operand] = vjp_value;
      auto genericOp = op->getParentOfType<linalg::GenericOp>();
      if (genericOp) {
        llvm::outs() << "outer arg: "
                     << (operand.getParentRegion() ==
                         &genericOp.getBodyRegion())
                     << "\n";
      }
    } else {
      env[operand] =
          rewriter.create<mlir::AddFOp>(op->getLoc(), env[operand], vjp_value);
    }
    op_index++;
  }
}

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

  // TODO: the topLevel flag here is poor API design, we need a way to determine
  // if the gradient signal should be passed in or automatically initialized
  // to 1.
  static FuncOp differentiateFunction(FuncOp funcOp, ArrayAttr gradientsOf,
                                      ConversionPatternRewriter &rewriter,
                                      bool topLevel = false) {
    Region *region = funcOp.getCallableRegion();
    auto *context = funcOp.getContext();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    if (!region) {
      funcOp->emitError("Function region cannot be null");
      return nullptr;
    }

    // Need to double check the return type.
    assert(funcOp.getType().getNumResults() == 1 &&
           "differentiating functions with more than one result not supported");
    if (!topLevel) {
      funcOp.insertArgument(funcOp.getNumArguments(),
                            funcOp.getType().getResult(0), {});
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
      auto opName = op->getName().getStringRef();
      if (opName == "std.return") {
        // This is the exit point
        rewriter.setInsertionPoint(op);
        Value operand = op->getOperand(0);
        // Initialize the gradient signal to 1.0
        if (topLevel) {
          env[operand] = onesLike(op->getLoc(), operand, rewriter);
        } else {
          env[operand] = funcOp.getArgument(funcOp.getNumArguments() - 1);
        }
        rewriter.eraseOp(op);
      } else if (opName == "std.cmpf") {
        continue;
      } else if (opName == "std.select") {
        auto selectOp = dyn_cast<mlir::SelectOp>(op);
        auto trueDefiningOp = selectOp.getTrueValue().getDefiningOp();
        llvm::DenseMap<Value, Value> selectEnv;
        std::queue<mlir::Operation *> frontier;
        if (trueDefiningOp) {
          frontier.push(trueDefiningOp);
        }

        auto falseDefiningOp = selectOp.getFalseValue().getDefiningOp();
        if (falseDefiningOp) {
          frontier.push(falseDefiningOp);
        }

        // Traverse the use chain here, computing derivatives.
        while (!frontier.empty()) {
          mlir::Operation *next = frontier.front();
          frontier.pop();
          populateVJP(next, selectEnv, rewriter);
          next->setAttr("visited",
                        mlir::BoolAttr::get(next->getContext(), true));
          for (auto operand : next->getOperands()) {
            if (operand.getDefiningOp()) {
              frontier.push(operand.getDefiningOp());
            }
          }
        }

        Value selectResult = rewriter.create<mlir::SelectOp>(
            selectOp.getLoc(), selectOp.getCondition(),
            selectEnv[selectOp.getTrueValue()],
            selectEnv[selectOp.getFalseValue()]);
        env[selectOp.getTrueValue()] = selectResult;
        env[selectOp.getFalseValue()] = selectResult;
      } else if (opName == "tensor.cast" || opName == "tensor.extract_slice" ||
                 opName == "memref.tensor_load") {
        // TODO: tensor.cast ops are currently only used for debugging.
        continue;
      } else {
        size_t op_index = 0;
        if (op->getNumResults() == 0) {
          continue;
        }

        for (Value operand : op->getOperands()) {
          // Compute the pullback (VJP).
          // TODO: Gotta be a better way to structure/abstract this. It's
          // essentially a huge switch statement on the operator name.
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
          } else if (opName == "math.exp") {
            assert(op->getNumResults() == 1 &&
                   "math.exp op did not have exactly 1 result");
            vjp_value = rewriter.create<mlir::MulFOp>(op->getLoc(), vjp_value,
                                                      op->getResult(0));
          } else if (opName == "math.log") {
            vjp_value = rewriter.create<mlir::DivFOp>(op->getLoc(), vjp_value,
                                                      op->getOperand(0));
          } else if (opName == "scf.if") {
            auto ifOp = dyn_cast<scf::IfOp>(op);
            vjp_value = reverseIfOp(ifOp, rewriter);
          } else if (opName == "linalg.generic") {
            auto genericOp = dyn_cast<linalg::GenericOp>(op);
            if (op_index > (size_t)genericOp.getNumInputs() - 1)
              continue;

            auto numIterators = genericOp.iterator_types().size();
            SmallVector<AffineMap, 6> indexing_maps(
                op->getNumOperands() + 1,
                rewriter.getMultiDimIdentityMap(numIterators));
            SmallVector<StringRef, 6> iterator_types(
                numIterators, getParallelIteratorTypeName());

            // Value output = env[operand]
            //                    ? env[operand]
            //                    : getZero(operand.getLoc(), operand,
            //                    rewriter);
            Value output = getZero(operand.getLoc(), operand, rewriter);
            auto outputShape = output.getType().dyn_cast<ShapedType>();
            assert(outputShape.hasRank() && "output must be a ranked type");
            auto generic_indexing_maps = genericOp.getIndexingMaps();
            auto op_count = genericOp.getNumOperands();
            SmallVector<Value> inputs;
            for (size_t i = 0; i < op_count; i++) {
              if (i == op_index) {
                indexing_maps[i] = generic_indexing_maps[i];
                inputs.push_back(op->getOperand(i));
              } else if (i == op_count - 1) {
                indexing_maps[i + 1] = generic_indexing_maps[op_index];
                // Add the gradient signal as an argument at the end of the
                // inputs.
                inputs.push_back(vjp_value);
                indexing_maps[i] = generic_indexing_maps[op_count - 1];
              } else {
                indexing_maps[i] = generic_indexing_maps[i];
                inputs.push_back(op->getOperand(i));
              }
            }

            llvm::DenseMap<Value, Value> bbEnv;
            llvm::SmallVector<Value> genericOperands;
            for (Value arg : genericOp.getBodyRegion().getArguments()) {
              genericOperands.push_back(arg);
            }
            // Region *newGenericRegion;

            auto adjoint = rewriter.create<linalg::GenericOp>(
                operand.getLoc(), /*resultTensorType=*/operand.getType(),
                /*inputs=*/inputs, /*outputs=*/ValueRange({output}),
                /*indexing_maps=*/indexing_maps,
                /*iterator_types=*/iterator_types,
                [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
                  SmallVector<mlir::Operation *> genericRegionOps =
                      cloneLinalgGenericBasicBlock(genericOp, builder,
                                                   regionArgs, genericOperands);

                  for (auto it = genericRegionOps.rbegin();
                       it != genericRegionOps.rend(); it++) {
                    auto rop = *it;
                    if (rop->getName().getStringRef() == "linalg.yield") {
                      bbEnv[rop->getOperand(0)] =
                          regionArgs[regionArgs.size() - 2];
                      rewriter.eraseOp(rop);
                      // newGenericRegion = rop->getParentRegion();
                    } else {
                      populateVJP(rop, bbEnv, builder);
                    }
                  }

                  // if (!bbEnv[regionArgs[op_index]]) {
                  //   bbEnv[regionArgs[op_index]] =
                  //       onesLike(loc, regionArgs[op_index], builder);
                  // }
                  // This add operation is required in the case of undoing
                  // reductions. It might be possible to omit this, if the
                  // output argument is never used in the primal, or perhaps if
                  // the primal iterator types do not include reductions.
                  Value add_res = builder.create<mlir::AddFOp>(
                      loc, bbEnv[regionArgs[op_index]],
                      regionArgs[regionArgs.size() - 1]);

                  builder.create<linalg::YieldOp>(loc, add_res);
                  // builder.create<linalg::YieldOp>(loc, regionArgs[0]);
                });
            vjp_value = adjoint.getResult(0);

            // Need to determine if this value was defined outside of the
            // generic op
            // for (auto pair : bbEnv) {
            //   assert(newGenericRegion && "newGenericRegion was not defined");
            //   if (pair.first.getParentRegion() != newGenericRegion) {
            //     if (env[pair.first]) {
            //       env[pair.first] = rewriter.create<mlir::AddFOp>(
            //           pair.first.getLoc(), env[pair.first], pair.second);
            //     } else {
            //       env[pair.first] = pair.second;
            //     }
            //   }
            // }
          } else if (opName == "linalg.dot") {
            if (op_index > 1)
              continue;

            SmallVector<AffineMap, 6> indexing_maps(
                op->getNumOperands(), rewriter.getMultiDimIdentityMap(1));
            indexing_maps[0] = indexing_maps[0].getSubMap({});
            indexing_maps[1] = indexing_maps[1].getSubMap({0});
            indexing_maps[2] = indexing_maps[2].getSubMap({0});
            auto library_call =
                op_index == 0 ? "sdot_grad_first" : "sdot_grad_second";
            auto adjoint = rewriter.create<linalg::GenericOp>(
                operand.getLoc(), /*resultTensorTypes=*/operand.getType(),
                /*inputs=*/
                ValueRange({vjp_value, op->getOperand(1 - op_index)}),
                /*outputs=*/ValueRange({operand}), indexing_maps,
                /*iteratorTypes=*/
                SmallVector<StringRef>({getParallelIteratorTypeName()}),
                /*doc=*/"Copy and scalar multiplication",
                /*library call=*/library_call,
                [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
                  Value mul_res = builder.create<mlir::MulFOp>(
                      loc, regionArgs[0], regionArgs[1]);
                  builder.create<linalg::YieldOp>(loc, mul_res);
                });
            vjp_value = adjoint.getResult(0);
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
                  /*doc=*/"Vector-vector outer product",
                  /*library call=*/"souter",
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
                  /*doc=*/"Vector-Matrix multiplication",
                  /*library call=*/"svecmat",
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
              indexingMaps[1] = indexingMaps[1].getSubMap({1});
              indexingMaps[2] = indexingMaps[2].getSubMap({0});
              SmallVector<StringRef, 6> iteratorTypes(
                  {getParallelIteratorTypeName(),
                   getReductionIteratorTypeName()});
              auto matmulOp = rewriter.create<linalg::GenericOp>(
                  operand.getLoc(),
                  /*resultTensorTypes=*/operand.getType(),
                  /*inputs=*/ValueRange({op->getOperand(1), vjp_value}),
                  /*outputs=*/ValueRange({zero}),
                  /*indexingMaps=*/indexingMaps,
                  /*iteratorTypes=*/iteratorTypes,
                  /*doc=*/"Matrix-vector multiplication",
                  /*library_call=*/"smatvec",
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
          } else if (opName == "linalg.matmul") {
            if (op_index > 1) {
              continue;
            }
            Value zero = env[operand]
                             ? env[operand]
                             : getZero(operand.getLoc(), operand, rewriter);
            SmallVector<AffineMap, 3> indexingMaps(
                op->getNumOperands(), rewriter.getMultiDimIdentityMap(3));
            if (op_index == 0) {
              indexingMaps[0] = indexingMaps[0].getSubMap({0, 1});
              indexingMaps[1] = indexingMaps[1].getSubMap({2, 1});
              indexingMaps[2] = indexingMaps[2].getSubMap({0, 2});
            } else {
              indexingMaps[0] = indexingMaps[0].getSubMap({1, 0});
              indexingMaps[1] = indexingMaps[1].getSubMap({1, 2});
              indexingMaps[2] = indexingMaps[2].getSubMap({0, 2});
            }
            SmallVector<StringRef, 6> iteratorTypes(
                {getParallelIteratorTypeName(), getReductionIteratorTypeName(),
                 getParallelIteratorTypeName()});
            SmallVector<Value> inputs(2);
            if (op_index == 0) {
              inputs[0] = vjp_value;
              inputs[1] = op->getOperand(1);
            } else {
              inputs[0] = op->getOperand(0);
              inputs[1] = vjp_value;
            }
            auto library_call =
                op_index == 0 ? "smatmul_grad_first" : "smatmul_grad_second";
            auto matmulOp = rewriter.create<linalg::GenericOp>(
                operand.getLoc(), operand.getType(), inputs, ValueRange({zero}),
                indexingMaps, iteratorTypes,
                /*doc=*/"Transposed matrix multiplication",
                /*library call=*/library_call,
                [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
                  Value mul_res = builder.create<mlir::MulFOp>(
                      loc, regionArgs[0], regionArgs[1]);
                  Value add_res =
                      builder.create<mlir::AddFOp>(loc, regionArgs[2], mul_res);
                  builder.create<linalg::YieldOp>(loc, add_res);
                });
            vjp_value = matmulOp.getResult(0);
          } else if (opName == "tensor.extract") {
            op->emitError("differentiating tensor.extract not yet supported");
          } else if (opName == "std.call") {
            auto callOp = dyn_cast<mlir::CallOp>(op);
            std::stringstream gradFuncStream;
            gradFuncStream << "__grad_" << callOp.callee().str() << "_arg"
                           << op_index;
            auto gradFuncName = gradFuncStream.str();
            auto dFuncOp =
                dyn_cast_or_null<FuncOp>(moduleOp.lookupSymbol(gradFuncName));
            if (!dFuncOp) {
              auto primalFunc =
                  dyn_cast<FuncOp>(moduleOp.lookupSymbol(callOp.calleeAttr()));
              dFuncOp =
                  copyFunctionDeclaration(primalFunc, gradFuncName, rewriter);

              auto innerGradsOf = ArrayAttr::get(
                  context,
                  {IntegerAttr::get(IntegerType::get(context, 64), op_index)});

              dFuncOp = differentiateFunction(dFuncOp, innerGradsOf, rewriter);
            }
            llvm::SmallVector<Value> operands(callOp.getOperands());
            operands.push_back(vjp_value);
            auto adjointCall = rewriter.create<mlir::CallOp>(callOp.getLoc(),
                                                             dFuncOp, operands);
            vjp_value = adjointCall.getResult(op_index);
          } else {
            llvm::outs() << "Unrecognized op: " << opName << "\n";
          }

          // Add the gradient signals.
          if (op->hasAttr("visited")) {
            // Do nothing
          } else if (!env[operand]) {
            env[operand] = vjp_value;
          } else {
            env[operand] = rewriter.create<mlir::AddFOp>(
                op->getLoc(), env[operand], vjp_value);
          }
          op_index++;
        }
      }
    }

    // for (auto it = ops.rbegin(); it != ops.rend(); it++) {
    //   Operation *op = *it;
    //   auto opName = op->getName().getStringRef();
    //   if (opName == "std.select") {
    //     auto selectOp = dyn_cast<mlir::SelectOp>(op);
    //     env[selectOp.getResult()] = rewriter.create<SelectOp>(
    //         selectOp.getLoc(), selectOp.getCondition(),
    //         env[selectOp.getTrueValue()], env[selectOp.getFalseValue()]);

    //     // TODO: double check gradient accumulation for correctness.
    //     env[selectOp.getTrueValue()] = env[selectOp.getResult()];
    //     env[selectOp.getFalseValue()] = env[selectOp.getResult()];
    //   }
    // }

    auto fntyp = funcOp.getType();
    // if (!env[region->getArgument(0)]) {
    //   env[region->getArgument(0)] = getZero(region->getArgument(0).getLoc(),
    //                                         region->getArgument(0),
    //                                         rewriter);
    // }
    SmallVector<Type> returnType(funcOp.getType().getNumInputs());
    SmallVector<Value> returnValue(funcOp.getType().getNumInputs());
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
      if (!env[region->getArgument(0)]) {
        funcOp.emitError("Gradient of first argument not found");
        return nullptr;
      }
      returnValue[0] = env[region->getArgument(0)];
    }
    funcOp.setType(
        FunctionType::get(funcOp.getContext(), fntyp.getInputs(), returnType));
    rewriter.create<mlir::ReturnOp>(region->getLoc(), returnValue);
    return funcOp;
  }

  static FuncOp copyFunctionDeclaration(FuncOp funcOp, llvm::StringRef funcName,
                                        ConversionPatternRewriter &rewriter) {
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointAfter(funcOp);
    auto newOp = static_cast<FuncOp>(rewriter.clone(*funcOp));

    newOp.setName(funcName);
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
      auto denseAttr = shapedType.getElementTypeBitWidth() == 32
                           ? DenseFPElementsAttr::get(shapedType, {0.0f})
                           : DenseFPElementsAttr::get(shapedType, {0.0});
      return rewriter.create<mlir::ConstantOp>(loc, denseAttr);
    }
    llvm_unreachable("not yet implemented");
    return nullptr;
  }

  static mlir::Value reverseIfOp(scf::IfOp ifOp,
                                 ConversionPatternRewriter &rewriter) {
    // TODO: This is incomplete. I need a procedure to collect free variables in
    // the thenBlock of the primal operation.
    auto adjointIf = rewriter.create<scf::IfOp>(
        ifOp->getLoc(), /*resultTypes=*/ifOp->getResult(0).getType(),
        /*cond=*/ifOp.condition(),
        /*thenBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          // TODO: Need to make a "reverse block" or "reverse region" function.
          llvm::SmallVector<Operation *> thenRegionOps;
          for (auto &thenRegionOp : ifOp.thenRegion().getOps()) {
            thenRegionOps.push_back(&thenRegionOp);
          }
          llvm::DenseMap<Value, Value> env;
          for (auto it = thenRegionOps.rbegin(); it != thenRegionOps.rend();
               it++) {
            auto op = *it;
            auto opName = op->getName().getStringRef();
            if (opName == "scf.yield") {
              Value operand = op->getOperand(0);
              // Initialize the gradient signal to 1.0
              env[operand] = builder.create<mlir::ConstantOp>(
                  loc, FloatAttr::get(operand.getType(), 1.0));
            } else {
              size_t op_index = 0;
              for (Value operand : op->getOperands()) {
                if (ifOp->getNumResults() == 0) {
                  ifOp->emitError("op had zero results");
                  return;
                }

                Value vjp_value = env[op->getResult(0)];
                if (opName == "std.mulf") {
                  vjp_value = rewriter.create<mlir::MulFOp>(
                      loc, vjp_value, op->getOperand(1 - op_index));
                } else {
                  llvm::outs()
                      << "(SCF if thenBuilder) unrecognized op: " << opName
                      << "\n";
                }

                // Add the gradient signals.
                if (!env[operand]) {
                  env[operand] = vjp_value;
                } else {
                  env[operand] = builder.create<mlir::AddFOp>(loc, env[operand],
                                                              vjp_value);
                }
                op_index++;
              }
            }
          }
          builder.create<scf::YieldOp>(
              loc, env[thenRegionOps.pop_back_val()->getOperand(0)]);
        },
        /*elseBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          Value cst = rewriter.create<mlir::ConstantOp>(
              loc, FloatAttr::get(ifOp->getResult(0).getType(), 0.0));
          builder.create<scf::YieldOp>(loc, cst);
        });
    return adjointIf.getResult(0);
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
  void getDependentDialects(DialectRegistry &registry) const override {}
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