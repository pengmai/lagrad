#include "Standalone/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
using namespace mlir;
SmallVector<mlir::Operation *>
cloneBasicBlock(llvm::iterator_range<Region::OpIterator> bbOps,
                OpBuilder &builder, ValueRange regionArgs,
                SmallVector<Value> bbOperands) {
  SmallVector<mlir::Operation *> newRegionOps;
  DenseMap<Value, Value> old_to_new;
  for (size_t i = 0; i < bbOperands.size(); i++) {
    // The last generic operand is shifted by one. It corresponds to the output
    // in the primal, but the gradient signal is inserted at the end of the
    // adjoint, hence the shift. This is also currently used with more ops than
    // linalg.generic.
    if (i == bbOperands.size() - 1) {
      old_to_new[bbOperands[i]] = regionArgs[bbOperands.size()];
    } else {
      old_to_new[bbOperands[i]] = regionArgs[i];
    }
  }

  for (auto &op : bbOps) {
    auto clonedOp = builder.clone(op);
    // Need to perform this old_to_new remapping for nested regions/blocks
    clonedOp->walk([&](Operation *nestedOp) {
      for (size_t i = 0; i < nestedOp->getNumOperands(); i++) {
        if (old_to_new[nestedOp->getOperand(i)]) {
          nestedOp->setOperand(i, old_to_new[nestedOp->getOperand(i)]);
        }
      }
    });
    for (size_t i = 0; i < clonedOp->getNumOperands(); i++) {
      assert(op.getNumResults() < 2 &&
             "basic block op with more than two results "
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
    newRegionOps.push_back(clonedOp);
  }

  return newRegionOps;
}

FuncOp copyFunctionDeclaration(FuncOp funcOp, llvm::StringRef funcName,
                               OpBuilder &rewriter) {
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointAfter(funcOp);
  auto newOp = static_cast<FuncOp>(rewriter.clone(*funcOp));

  newOp.setName(funcName);
  return newOp;
}

FuncOp differentiateFunction(FuncOp funcOp, ArrayAttr gradientsOf,
                             ConversionPatternRewriter &rewriter,
                             bool topLevel = false) {
  Region *region = funcOp.getCallableRegion();
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
    } else if (opName == "arith.cmpf") {
      continue;
    } else if (opName == "scf.if") {
      auto ifOp = dyn_cast<scf::IfOp>(op);
      assert(ifOp.getNumResults() == 1 &&
             "if ops with num results != 1 not yet supported");
      auto vjp_value = env[ifOp.getResult(0)];
      if (!vjp_value) {
        Value result = ifOp.getResult(0);
        vjp_value = onesLike(result.getLoc(), result, rewriter);
        env[result] = vjp_value;
      }

      // Collect the free variables in the then block of the if op
      SmallVector<Value> freeOperands;
      collectFreeVars(ifOp.thenBlock(), ifOp.thenRegion(), freeOperands);
      collectFreeVars(ifOp.elseBlock(), ifOp.elseRegion(), freeOperands);

      for (auto freeOperand : freeOperands) {
        auto result = reverseIfOp(ifOp, freeOperand, vjp_value, env, rewriter);
        if (!env[freeOperand]) {
          env[freeOperand] = result;
        } else {
          env[freeOperand] = rewriter.create<arith::AddFOp>(
              freeOperand.getLoc(), result, env[freeOperand]);
        }
      }
    } else if (opName == "tensor.cast") {
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
        if (opName == "arith.mulf") {
          vjp_value = rewriter.create<arith::MulFOp>(
              op->getLoc(), vjp_value, op->getOperand(1 - op_index));
        } else if (opName == "arith.addf") {
          // This has no effect on the VJP
        } else if (opName == "arith.subf") {
          if (op_index == 1) {
            vjp_value = rewriter.create<arith::NegFOp>(op->getLoc(), vjp_value);
          }
        } else if (opName == "arith.divf") {
          if (op_index == 0) {
            vjp_value = rewriter.create<arith::DivFOp>(op->getLoc(), vjp_value,
                                                       op->getOperand(1));
          } else {
            assert(op_index == 1 && "arith.divf op had more than 2 args");
            vjp_value = rewriter.create<arith::MulFOp>(op->getLoc(), vjp_value,
                                                       op->getOperand(0));
            vjp_value = rewriter.create<arith::NegFOp>(op->getLoc(), vjp_value);
            Value denom =
                rewriter.create<arith::MulFOp>(op->getLoc(), operand, operand);
            vjp_value =
                rewriter.create<arith::DivFOp>(op->getLoc(), vjp_value, denom);
          }
        } else if (opName == "arith.negf") {
          vjp_value = rewriter.create<arith::NegFOp>(op->getLoc(), vjp_value);
        } else if (opName == "math.exp") {
          assert(op->getNumResults() == 1 &&
                 "math.exp op did not have exactly 1 result");
          vjp_value = rewriter.create<arith::MulFOp>(op->getLoc(), vjp_value,
                                                     op->getResult(0));
        } else if (opName == "math.log") {
          vjp_value = rewriter.create<arith::DivFOp>(op->getLoc(), vjp_value,
                                                     op->getOperand(0));
        } else if (opName == "linalg.generic") {
          auto genericOp = dyn_cast<linalg::GenericOp>(op);
          if (op_index > static_cast<size_t>(genericOp.getNumInputs() - 1))
            continue;

          // Additionally compute adjoints for all free variables. We only want
          // this to run once, hence the if op_index == 0.
          if (op_index == 0) {
            SmallVector<Value> freeOperands;
            collectFreeVars(genericOp.getBody(), genericOp.getBodyRegion(),
                            freeOperands);
            for (auto freeOperand : freeOperands) {
              if (!freeOperand.getType().isa<FloatType>()) {
                continue;
              }
              // Not totally sure if we can use the VJP value as-is, watch out
              // for bugs.
              auto out = reverseGenericOp(genericOp, freeOperand, vjp_value, -1,
                                          rewriter);
              auto result =
                  rewriter.create<tensor::ExtractOp>(freeOperand.getLoc(), out);
              if (!env[freeOperand]) {
                env[freeOperand] = result;
              } else {
                env[freeOperand] = rewriter.create<arith::AddFOp>(
                    freeOperand.getLoc(), result, env[freeOperand]);
              }
            }
          }
          vjp_value = reverseGenericOp(genericOp, operand, vjp_value, op_index,
                                       rewriter);
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
                Value mul_res = builder.create<arith::MulFOp>(
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
                /*outputs=*/ValueRange({operand}), indexingMaps, iteratorTypes,
                /*doc=*/"Vector-vector outer product",
                /*library call=*/"souter",
                [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
                  Value mul_res = builder.create<arith::MulFOp>(
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
                  auto mul_res = builder.create<arith::MulFOp>(
                      loc, regionArgs[0], regionArgs[1]);
                  auto reduced = builder.create<arith::AddFOp>(loc, mul_res,
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
                  auto mul_res = builder.create<arith::MulFOp>(
                      loc, regionArgs[0], regionArgs[1]);
                  auto reduced = builder.create<arith::AddFOp>(loc, mul_res,
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
                /*outputs=*/ValueRange({operand}), indexingMaps, iteratorTypes,
                [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
                  Value mul_res = builder.create<arith::MulFOp>(
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
                Value mul_res = builder.create<arith::MulFOp>(
                    loc, regionArgs[0], regionArgs[1]);
                Value add_res =
                    builder.create<arith::AddFOp>(loc, regionArgs[2], mul_res);
                builder.create<linalg::YieldOp>(loc, add_res);
              });
          vjp_value = matmulOp.getResult(0);
        } else if (opName == "tensor.extract") {
          if (op_index > 0) {
            continue;
          }
          vjp_value = reverseTensorExtractOp(dyn_cast<tensor::ExtractOp>(op),
                                             operand, vjp_value, rewriter);
        } else if (opName == "tensor.extract_slice") {
          auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op);
          auto space = getZero(operand.getLoc(), operand, rewriter);
          vjp_value = rewriter.create<tensor::InsertSliceOp>(
              operand.getLoc(), space.getType(), vjp_value, space,
              extractSliceOp.offsets(), extractSliceOp.sizes(),
              extractSliceOp.strides(), extractSliceOp.static_offsets(),
              extractSliceOp.static_sizes(), extractSliceOp.static_strides());
        } else if (opName == "std.call") {
          vjp_value = reverseCallOp(dyn_cast<mlir::CallOp>(op), moduleOp,
                                    vjp_value, op_index, rewriter);
        } else {
          llvm::outs() << "Unrecognized op: " << opName << "\n";
        }

        // Add the gradient signals.
        if (op->hasAttr("visited")) {
          // Do nothing
        } else if (!env[operand]) {
          env[operand] = vjp_value;
        } else {
          env[operand] = rewriter.create<arith::AddFOp>(
              op->getLoc(), env[operand], vjp_value);
        }
        op_index++;
      }
    }
  }

  auto fntyp = funcOp.getType();
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
Value onesLike(Location loc, Value operand, OpBuilder &builder) {
  if (operand.getType().isa<FloatType>()) {
    return builder.create<arith::ConstantOp>(
        loc, FloatAttr::get(operand.getType(), 1.0));
  }
  if (operand.getType().isa<ShapedType>()) {
    auto shapedType = operand.getType().dyn_cast<ShapedType>();
    auto denseAttr = shapedType.getElementTypeBitWidth() == 32
                         ? DenseFPElementsAttr::get(shapedType, {1.0f})
                         : DenseFPElementsAttr::get(shapedType, {1.0});
    return builder.create<arith::ConstantOp>(loc, denseAttr);
  }
  llvm::outs() << "ones for type " << operand.getType() << " not implemented\n";
  llvm_unreachable("");
  return nullptr;
}

Value getZero(Location loc, Value operand, OpBuilder &rewriter) {
  if (operand.getType().isa<FloatType>()) {
    return rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(operand.getType(), 0.0));
  }
  if (operand.getType().isa<ShapedType>()) {
    auto shapedType = operand.getType().dyn_cast<ShapedType>();
    // Will automatically be broadcasted to the right shape.
    auto denseAttr = shapedType.getElementTypeBitWidth() == 32
                         ? DenseFPElementsAttr::get(shapedType, {0.0f})
                         : DenseFPElementsAttr::get(shapedType, {0.0});
    return rewriter.create<arith::ConstantOp>(loc, denseAttr);
  }
  llvm_unreachable("not yet implemented");
  return nullptr;
}

Value constLike(Location loc, Value operand, double scalar,
                OpBuilder &builder) {
  if (operand.getType().isa<FloatType>()) {
    return builder.create<arith::ConstantOp>(
        loc, FloatAttr::get(operand.getType(), scalar));
  }
  if (operand.getType().isa<ShapedType>()) {
    auto shapedType = operand.getType().dyn_cast<ShapedType>();
    auto denseAttr =
        shapedType.getElementTypeBitWidth() == 32
            ? DenseFPElementsAttr::get(shapedType, {static_cast<float>(scalar)})
            : DenseFPElementsAttr::get(shapedType, {scalar});
    return builder.create<arith::ConstantOp>(loc, denseAttr);
  }
  llvm::outs() << "scalar for type " << operand.getType()
               << " not implemented\n";
  llvm_unreachable("");
  return nullptr;
}

void collectFreeVars(Block *parentBlock, Region &region,
                     SmallVector<Value> &out) {
  region.walk([&](Operation *regionOp) {
    for (auto operand : regionOp->getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (dyn_cast_or_null<arith::ConstantOp>(definingOp)) {
        continue;
      }
      if (operand.getParentBlock() != parentBlock &&
          (std::find(out.begin(), out.end(), operand) == out.end())) {
        out.push_back(operand);
      }
    }
  });
}

void populateVJP(Operation *op, ModuleOp moduleOp,
                 llvm::DenseMap<Value, Value> &env,
                 ConversionPatternRewriter &rewriter) {
  auto opName = op->getName().getStringRef();
  if (opName == "arith.sitofp") {
    // The input is an integer so can't have a gradient signal.
    return;
  }
  if (opName == "scf.if") {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    assert(ifOp.getNumResults() == 1 &&
           "if ops with num results != 1 not yet supported");
    auto vjp_value = env[ifOp.getResult(0)];
    if (!vjp_value) {
      Value result = ifOp.getResult(0);
      vjp_value = onesLike(result.getLoc(), result, rewriter);
      env[result] = vjp_value;
    }

    // Collect the free variables in the then block of the if op
    SmallVector<Value> freeOperands;
    collectFreeVars(ifOp.thenBlock(), ifOp.thenRegion(), freeOperands);
    collectFreeVars(ifOp.elseBlock(), ifOp.elseRegion(), freeOperands);

    for (auto freeOperand : freeOperands) {
      auto result = reverseIfOp(ifOp, freeOperand, vjp_value, env, rewriter);
      if (!env[freeOperand]) {
        env[freeOperand] = result;
      } else {
        env[freeOperand] = rewriter.create<arith::AddFOp>(
            freeOperand.getLoc(), result, env[freeOperand]);
      }
    }
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

    if (opName == "arith.mulf") {
      vjp_value = rewriter.create<arith::MulFOp>(op->getLoc(), vjp_value,
                                                 op->getOperand(1 - op_index));
    } else if (opName == "arith.addf") {
      // This has no effect on the VJP
    } else if (opName == "arith.subf") {
      if (op_index == 1) {
        vjp_value = rewriter.create<arith::NegFOp>(op->getLoc(), vjp_value);
      }
    } else if (opName == "arith.divf") {
      if (op_index == 0) {
        vjp_value = rewriter.create<arith::DivFOp>(op->getLoc(), vjp_value,
                                                   op->getOperand(1));
      } else {
        vjp_value = rewriter.create<arith::MulFOp>(op->getLoc(), vjp_value,
                                                   op->getOperand(0));
        vjp_value = rewriter.create<arith::NegFOp>(op->getLoc(), vjp_value);
        Value denom =
            rewriter.create<arith::MulFOp>(op->getLoc(), operand, operand);
        vjp_value =
            rewriter.create<arith::DivFOp>(op->getLoc(), vjp_value, denom);
      }
    } else if (opName == "std.select") {
      // auto selectOp = dyn_cast<mlir::SelectOp>(op);
      llvm_unreachable("std.select not yet implemented");
    } else if (opName == "math.exp") {
      assert(op->getNumResults() == 1 &&
             "math.exp op did not have exactly 1 result");
      vjp_value = rewriter.create<arith::MulFOp>(op->getLoc(), vjp_value,
                                                 op->getResult(0));
    } else if (opName == "math.sin") {
      auto cos = rewriter.create<math::CosOp>(op->getLoc(), operand);
      vjp_value = rewriter.create<arith::MulFOp>(op->getLoc(), cos, vjp_value);
    } else if (opName == "math.cos") {
      auto sin = rewriter.create<math::SinOp>(op->getLoc(), vjp_value);
      vjp_value = rewriter.create<arith::MulFOp>(op->getLoc(), sin, vjp_value);
      vjp_value = rewriter.create<arith::NegFOp>(op->getLoc(), vjp_value);
    } else if (opName == "math.sqrt") {
      auto half = constLike(op->getLoc(), operand, 0.5, rewriter);
      vjp_value = rewriter.create<arith::MulFOp>(op->getLoc(), vjp_value, half);
      // This is a bit of a math trick. Note the result is sqrt(operand)
      vjp_value = rewriter.create<arith::DivFOp>(op->getLoc(), vjp_value,
                                                 op->getResult(0));
    } else if (opName == "std.call") {
      vjp_value = reverseCallOp(dyn_cast<CallOp>(op), moduleOp, vjp_value,
                                op_index, rewriter);
    } else if (opName == "tensor.extract") {
      if (op_index > 0) {
        continue;
      }
      vjp_value = reverseTensorExtractOp(dyn_cast<tensor::ExtractOp>(op),
                                         operand, vjp_value, rewriter);
    } else if (opName == "linalg.generic") {
      auto genericOp = dyn_cast<linalg::GenericOp>(op);
      if (op_index > static_cast<size_t>(genericOp.getNumInputs() - 1))
        continue;

      // Additionally compute adjoints for all free variables. We only want
      // this to run once, hence the if op_index == 0.
      if (op_index == 0) {
        SmallVector<Value> freeOperands;
        collectFreeVars(genericOp.getBody(), genericOp.getBodyRegion(),
                        freeOperands);
        for (auto freeOperand : freeOperands) {
          if (!freeOperand.getType().isa<FloatType>()) {
            continue;
          }
          // Not totally sure if we can use the VJP value as-is, watch out
          // for bugs.
          auto out =
              reverseGenericOp(genericOp, freeOperand, vjp_value, -1, rewriter);
          auto result =
              rewriter.create<tensor::ExtractOp>(freeOperand.getLoc(), out);
          if (!env[freeOperand]) {
            env[freeOperand] = result;
          } else {
            env[freeOperand] = rewriter.create<arith::AddFOp>(
                freeOperand.getLoc(), result, env[freeOperand]);
          }
        }
      }
      vjp_value =
          reverseGenericOp(genericOp, operand, vjp_value, op_index, rewriter);
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
            Value mul_res = builder.create<arith::MulFOp>(loc, regionArgs[0],
                                                          regionArgs[1]);
            builder.create<linalg::YieldOp>(loc, mul_res);
          });
      vjp_value = adjoint.getResult(0);
    } else {
      llvm::outs() << "(populateVJP) unrecognized op: " << opName << "\n";
    }

    // Add the gradient signals.
    if (!env[operand]) {
      env[operand] = vjp_value;
    } else {
      env[operand] =
          rewriter.create<arith::AddFOp>(op->getLoc(), env[operand], vjp_value);
    }
    op_index++;
  }
}

Value reverseGenericOp(linalg::GenericOp op, Value operand, Value vjp_value,
                       int op_index, ConversionPatternRewriter &rewriter) {
  // Need to ensure:
  // if (op_index > (size_t)genericOp.getNumInputs() - 1)
  //   continue;
  auto numIterators = op.iterator_types().size();
  SmallVector<AffineMap, 6> indexing_maps(
      op->getNumOperands() + 1, rewriter.getMultiDimIdentityMap(numIterators));
  SmallVector<StringRef, 6> iterator_types(numIterators,
                                           getParallelIteratorTypeName());

  Value output;
  if (op_index == -1) {
    auto zeroDTensorType = RankedTensorType::get({}, operand.getType());
    auto denseFPAttr = operand.getType().getIntOrFloatBitWidth() == 32
                           ? DenseFPElementsAttr::get(zeroDTensorType, {0.0f})
                           : DenseFPElementsAttr::get(zeroDTensorType, {0.0});
    output = rewriter.create<arith::ConstantOp>(operand.getLoc(), denseFPAttr);
  } else {
    output = getZero(operand.getLoc(), operand, rewriter);
  }
  auto outputShape = output.getType().dyn_cast<ShapedType>();
  assert(outputShape.hasRank() && "output must be a ranked type");
  auto generic_indexing_maps = op.getIndexingMaps();
  auto op_count = op.getNumOperands();
  SmallVector<Value> inputs;
  for (size_t i = 0; i < op_count; i++) {
    if (i == static_cast<size_t>(op_index)) {
      indexing_maps[i] = generic_indexing_maps[i];
      inputs.push_back(op.getOperand(i));
    } else if (i == op_count - 1) {
      if (op_index == -1) {
        // In the case of free variables, the output is assumed to be 0d.
        indexing_maps[i + 1] = indexing_maps[i + 1].getSubMap({});
      } else {
        // The output has to map the shape of the current argument.
        indexing_maps[i + 1] = generic_indexing_maps[op_index];
      }
      // Add the gradient signal as an argument at the end of the
      // inputs.
      inputs.push_back(vjp_value);
      indexing_maps[i] = generic_indexing_maps[op_count - 1];
    } else {
      indexing_maps[i] = generic_indexing_maps[i];
      inputs.push_back(op.getOperand(i));
    }
  }

  DenseMap<Value, Value> bbEnv;
  SmallVector<Value> genericOperands;
  for (Value arg : op.getBodyRegion().getArguments()) {
    genericOperands.push_back(arg);
  }

  auto adjoint = rewriter.create<linalg::GenericOp>(
      operand.getLoc(), /*resultTensorType=*/outputShape,
      /*inputs=*/inputs, /*outputs=*/ValueRange({output}),
      /*indexing_maps=*/indexing_maps,
      /*iterator_types=*/iterator_types,
      [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
        PatternRewriter::InsertionGuard insertionGuard(rewriter);
        SmallVector<mlir::Operation *> genericRegionOps =
            cloneBasicBlock(op.getOps(), builder, regionArgs, genericOperands);

        for (auto it = genericRegionOps.rbegin(); it != genericRegionOps.rend();
             it++) {
          auto rop = *it;
          if (rop->getName().getStringRef() == "linalg.yield") {
            bbEnv[rop->getOperand(0)] = regionArgs[regionArgs.size() - 2];
            rewriter.setInsertionPointAfter(rop);
            rewriter.eraseOp(rop);
          } else if (rop->getName().getStringRef() == "arith.cmpf") {
            continue;
          } else {
            populateVJP(rop, op->getParentOfType<ModuleOp>(), bbEnv, rewriter);
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
        auto new_operand = op_index == -1 ? operand : regionArgs[op_index];
        if (!bbEnv[new_operand]) {
          rewriter.create<linalg::YieldOp>(loc,
                                           getZero(loc, new_operand, rewriter));
        } else {
          Value add_res = rewriter.create<arith::AddFOp>(
              loc, bbEnv[new_operand], regionArgs[regionArgs.size() - 1]);

          rewriter.create<linalg::YieldOp>(loc, add_res);
        }
      });
  return adjoint.getResult(0);
}

Value reverseIfOp(scf::IfOp ifOp, Value freeOperand, Value vjp_value,
                  DenseMap<Value, Value> outer_env,
                  ConversionPatternRewriter &rewriter) {
  auto reverseIfBlock = [&](Region &ifRegion) {
    return [&](OpBuilder &builder, Location loc) {
      PatternRewriter::InsertionGuard insertionGuard(rewriter);
      auto primalRegionOps =
          cloneBasicBlock(ifRegion.getOps(), builder, {}, {});
      DenseMap<Value, Value> env;
      for (auto it = primalRegionOps.rbegin(); it != primalRegionOps.rend();
           it++) {
        auto op = *it;
        auto opName = op->getName().getStringRef();
        if (opName == "scf.yield") {
          Value operand = op->getOperand(0);
          env[operand] = vjp_value;
          rewriter.setInsertionPointAfter(op);
          rewriter.eraseOp(op);
        } else {
          populateVJP(op, ifOp->getParentOfType<ModuleOp>(), env, rewriter);
        }
      }
      // The free operand might only appear in one block but not the other.
      if (!env[freeOperand]) {
        rewriter.create<scf::YieldOp>(loc, getZero(loc, freeOperand, rewriter));
      } else {
        rewriter.create<scf::YieldOp>(loc, env[freeOperand]);
      }
    };
  };

  auto adjointIf = rewriter.create<scf::IfOp>(
      ifOp->getLoc(), /*resultTypes=*/freeOperand.getType(),
      /*cond=*/ifOp.condition(),
      /*thenBuilder=*/reverseIfBlock(ifOp.thenRegion()),
      /*elseBuilder=*/reverseIfBlock(ifOp.elseRegion()));
  return adjointIf.getResult(0);
}

Value reverseTensorExtractOp(tensor::ExtractOp op, Value operand,
                             Value vjp_value, OpBuilder &builder) {
  // Using a constant tensor is causing issues here. We need to
  // explicitly allocate a space using init_tensor.
  auto tensorType = op.tensor().getType().dyn_cast<ShapedType>();
  assert(tensorType.hasStaticShape() &&
         "only static shapes are currently supported");
  auto zero = getZero(operand.getLoc(), op.result(), builder);
  auto space = builder.create<linalg::InitTensorOp>(
      operand.getLoc(), ValueRange{}, tensorType.getShape(),
      op.result().getType());
  auto filled = builder.create<linalg::FillOp>(operand.getLoc(), zero, space);
  return builder.create<tensor::InsertOp>(op.getLoc(), vjp_value,
                                          filled.getResult(0), op.indices());
}

Value reverseCallOp(CallOp op, ModuleOp moduleOp, Value vjp_value,
                    size_t op_index, ConversionPatternRewriter &rewriter) {
  auto *context = op.getContext();
  std::stringstream gradFuncStream;
  gradFuncStream << "__grad_" << op.callee().str() << "_arg" << op_index;
  auto gradFuncName = gradFuncStream.str();
  auto dFuncOp = dyn_cast_or_null<FuncOp>(moduleOp.lookupSymbol(gradFuncName));
  if (!dFuncOp) {
    auto primalFunc = dyn_cast<FuncOp>(moduleOp.lookupSymbol(op.calleeAttr()));
    dFuncOp = copyFunctionDeclaration(primalFunc, gradFuncName, rewriter);

    auto innerGradsOf = ArrayAttr::get(
        context, {IntegerAttr::get(IntegerType::get(context, 64), op_index)});

    dFuncOp = differentiateFunction(dFuncOp, innerGradsOf, rewriter);
  }
  llvm::SmallVector<Value> operands(op.getOperands());
  operands.push_back(vjp_value);
  auto adjointCall =
      rewriter.create<mlir::CallOp>(op.getLoc(), dFuncOp, operands);
  assert(adjointCall.getNumResults() == 1 &&
         "expected adjoint call to produce 1 result");
  return adjointCall.getResult(0);
}

} // namespace mlir
