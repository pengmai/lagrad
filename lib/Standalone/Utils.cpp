#include "Standalone/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
using namespace mlir;

bool isFloatOrFloatTensor(Type typ) {
  return typ.isa<FloatType>() ||
         (typ.isa<RankedTensorType>() &&
          typ.dyn_cast<RankedTensorType>().getElementType().isa<FloatType>());
}

bool isIntOrIntTensor(Type typ) {
  return typ.isIntOrIndex() ||
         (typ.isa<RankedTensorType>() &&
          typ.dyn_cast<RankedTensorType>().getElementType().isIntOrIndex());
}

SmallVector<mlir::Operation *>
cloneBasicBlock(llvm::iterator_range<Region::OpIterator> bbOps,
                OpBuilder &builder, ValueRange regionArgs,
                SmallVector<Value> bbOperands, bool offsetInputs = true) {
  SmallVector<mlir::Operation *> newRegionOps;
  DenseMap<Value, Value> old_to_new;
  for (size_t i = 0; i < bbOperands.size(); i++) {
    // The last generic operand is shifted by one. It corresponds to the output
    // in the primal, but the gradient signal is inserted at the end of the
    // adjoint, hence the shift. This is also currently used with more ops than
    // linalg.generic.
    if (offsetInputs && i == bbOperands.size() - 1) {
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
      // We assume that region arguments and intermediate values will populate
      // this map. If an entry is missing, it should have been defined outside
      // the linalg.generic body.
      if (old_to_new[clonedOp->getOperand(i)]) {
        clonedOp->setOperand(i, old_to_new[clonedOp->getOperand(i)]);
      }
      for (size_t j = 0; j < op.getNumResults(); j++) {
        old_to_new[op.getResult(j)] = clonedOp->getResult(j);
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
  llvm::SmallDenseSet<Value> active;

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  for (auto it = ops.rbegin(); it != ops.rend(); it++) {
    Operation *op = *it;
    auto opName = op->getName().getStringRef();
    if (opName == "std.return") {
      // This is the exit point
      runActivityAnalysis(op, funcOp.getArguments(), active);
      rewriter.setInsertionPoint(op);
      assert(op->getNumOperands() == 1 &&
             "Expected function to return 1 value");
      Value operand = op->getOperand(0);
      // Initialize the gradient signal to 1.0
      if (topLevel) {
        env[operand] = onesLike(op->getLoc(), operand, rewriter, /*init=*/true);
      } else {
        env[operand] = funcOp.getArgument(funcOp.getNumArguments() - 1);
      }
      rewriter.eraseOp(op);
    } else if (opName == "arith.cmpf") {
      continue;
    } else if (op->getNumResults() != 0) {
      // &&
      //   llvm::any_of(op->getResults(),
      //                [&](OpResult res) { return active.contains(res); })) {
      populateVJP(op, moduleOp, env, rewriter);
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
      if (!returnValue[i]) {
        // Int tensors aren't being filtered out for some reason. This is a
        // temporary bandaid.
        auto argument = region->getArgument(argIndex);
        returnValue[i] =
            getZero(argument.getLoc(), argument, rewriter, /*init=*/false);
      }
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

Value onesLike(Location loc, Value operand, OpBuilder &builder,
               bool init = false) {
  if (operand.getType().isa<FloatType>()) {
    return builder.create<arith::ConstantOp>(
        loc, FloatAttr::get(operand.getType(), 1.0));
  }
  if (operand.getType().isa<ShapedType>()) {
    auto shapedType = operand.getType().dyn_cast<ShapedType>();
    assert(shapedType.getElementType().isa<FloatType>() &&
           "Shaped type was not a float type");
    if (init) {
      auto floatVal = shapedType.getElementTypeBitWidth() == 32 ? APFloat(1.0f)
                                                                : APFloat(1.0);
      auto one = builder.create<arith::ConstantFloatOp>(
          loc, floatVal, shapedType.getElementType().dyn_cast<FloatType>());
      auto space = builder.create<linalg::InitTensorOp>(
          loc, shapedType.getShape(), shapedType.getElementType());
      auto filled = builder.create<linalg::FillOp>(loc, one, space);
      return filled.getResult(0);
    }
    auto denseAttr = shapedType.getElementTypeBitWidth() == 32
                         ? DenseFPElementsAttr::get(shapedType, {1.0f})
                         : DenseFPElementsAttr::get(shapedType, {1.0});
    return builder.create<arith::ConstantOp>(loc, denseAttr);
  }
  llvm::outs() << "ones for type " << operand.getType() << " not implemented\n";
  llvm_unreachable("");
  return nullptr;
}

Value getZero(Location loc, Value operand, OpBuilder &rewriter,
              bool init = false) {
  if (operand.getType().isa<FloatType>()) {
    return rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(operand.getType(), 0.0));
  }
  if (operand.getType().isa<ShapedType>()) {
    auto shapedType = operand.getType().dyn_cast<ShapedType>();
    if (init) {
      auto zero = rewriter.create<arith::ConstantOp>(
          loc, FloatAttr::get(shapedType.getElementType(), 0.0));
      auto space = rewriter.create<linalg::InitTensorOp>(
          loc, shapedType.getShape(), shapedType.getElementType());
      auto filled = rewriter.create<linalg::FillOp>(loc, zero, space);
      return filled.getResult(0);
    } else {
      // Will automatically be broadcasted to the right shape.
      auto denseAttr = shapedType.getElementTypeBitWidth() == 32
                           ? DenseFPElementsAttr::get(shapedType, {0.0f})
                           : DenseFPElementsAttr::get(shapedType, {0.0});
      return rewriter.create<arith::ConstantOp>(loc, denseAttr);
    }
  }
  llvm::outs() << "getZero for type " << operand.getType()
               << " not yet implemented\n";
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
                     llvm::SmallDenseSet<Value> &out) {
  for (auto &op : region.getOps()) {
    for (auto operand : op.getOperands()) {
      if (!isFloatOrFloatTensor(operand.getType())) {
        continue;
      }
      auto definingOp = operand.getDefiningOp();
      if (dyn_cast_or_null<arith::ConstantOp>(definingOp)) {
        continue;
      }
      if (operand.getParentBlock() != parentBlock && !out.contains(operand)) {
        out.insert(operand);
      }
    }
    for (auto &childRegion : op.getRegions()) {
      for (auto &childBlock : childRegion.getBlocks()) {
        collectFreeVars(&childBlock, childRegion, out);
      }
    }
  }
}

// This is definitely a bandaid behind an explosion of complexity in the
// autodiff method.
void eraseUnusedCalls(ModuleOp moduleOp, PatternRewriter &rewriter) {
  moduleOp.walk([&](CallOp callOp) {
    if (callOp.calleeAttr().getValue().startswith("__grad") &&
        callOp.use_empty()) {
      // rewriter.eraseOp(callOp);
      // llvm::outs() << "saw callOp " << callOp.calleeAttr().getValue()
      //              << " with " << callOp.use_empty() << " uses\n";
    }
  });
}

Value reverseBatchMatmul(Operation *op, Value operand, Value vjp_value,
                         size_t op_index, ConversionPatternRewriter &rewriter) {
  auto bmmOp = dyn_cast<linalg::BatchMatmulOp>(op);
  size_t op_rank = 4;
  auto idMap = rewriter.getMultiDimIdentityMap(op_rank);
  auto zero = getZero(operand.getLoc(), operand, rewriter);
  assert(op_index < 2 && "Invalid operand index for batch matmul");
  auto indexingMaps =
      op_index == 0 ? SmallVector<AffineMap, 6>{idMap.getSubMap({0, 1, 2}),
                                                idMap.getSubMap({0, 3, 2}),
                                                idMap.getSubMap({0, 1, 3})}
                    : SmallVector<AffineMap, 6>{idMap.getSubMap({0, 2, 1}),
                                                idMap.getSubMap({0, 2, 3}),
                                                idMap.getSubMap({0, 1, 3})};
  SmallVector<StringRef, 6> iteratorTypes(
      {getParallelIteratorTypeName(), getParallelIteratorTypeName(),
       getReductionIteratorTypeName(), getParallelIteratorTypeName()});
  auto inputs = op_index == 0
                    ? SmallVector<Value, 2>{vjp_value, bmmOp.getOperand(1)}
                    : SmallVector<Value, 2>{bmmOp.getOperand(0), vjp_value};
  auto adjoint = rewriter.create<linalg::GenericOp>(
      operand.getLoc(),
      /*resultTensorTypes=*/operand.getType(),
      /*inputs=*/inputs,
      /*outputs=*/ValueRange({zero}), indexingMaps, iteratorTypes,
      [](OpBuilder &builder, Location loc, ValueRange regionArgs) {
        auto mul =
            builder.create<arith::MulFOp>(loc, regionArgs[0], regionArgs[1]);
        auto add = builder.create<arith::AddFOp>(loc, mul, regionArgs[2]);
        builder.create<linalg::YieldOp>(loc, add.getResult());
      });
  return adjoint.getResult(0);
}

void populateVJP(Operation *op, ModuleOp moduleOp,
                 llvm::DenseMap<Value, Value> &env,
                 ConversionPatternRewriter &rewriter) {
  auto opName = op->getName().getStringRef();
  if (opName == "arith.sitofp") {
    // The input is an integer so can't have a gradient signal.
    return;
  }
  if (opName == "linalg.fill") {
    // We ignore fill ops for now because we assume they don't propagate
    // gradient signal.
    return;
  }
  if (opName == "scf.if") {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (ifOp.getNumResults() == 0) {
      return;
    }
    assert(ifOp.getNumResults() == 1 &&
           "if ops with num results != 1 not yet supported");
    auto vjp_value = env[ifOp.getResult(0)];
    if (!vjp_value) {
      Value result = ifOp.getResult(0);
      vjp_value = onesLike(result.getLoc(), result, rewriter);
      env[result] = vjp_value;
    }

    // Collect the free variables in the then block of the if op
    llvm::SmallDenseSet<Value> freeOperands;
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
  } else if (opName == "scf.for") {
    auto forOp = dyn_cast<scf::ForOp>(op);
    assert(forOp.getNumResults() > 0 &&
           "for op with zero results not supported");
    size_t result_idx = -1;
    for (size_t idx = 0; idx < forOp.getNumResults(); idx++) {
      if (isFloatOrFloatTensor(forOp.getResult(idx).getType())) {
        result_idx = idx;
        break;
      }
    }

    Value result = forOp.getResult(result_idx);
    auto vjp_value = env[result];
    env[forOp.getIterOperands()[result_idx]] = env[result];
    assert(vjp_value && "vjp value for scf.for op was not found");
    llvm::SmallDenseSet<Value> freeOperands;
    collectFreeVars(forOp.getBody(), forOp.getLoopBody(), freeOperands);

    auto free_operand_vec = llvm::to_vector<4>(freeOperands);
    auto vjp_values = reverseForOp(forOp, free_operand_vec, vjp_value,
                                   result_idx, env, rewriter);
    for (auto result_pair : llvm::zip(free_operand_vec, vjp_values)) {
      auto free_operand = std::get<0>(result_pair);
      auto result_vjp = std::get<1>(result_pair);
      // If the free operand already has a space in the gradient, the for op
      // will add to that space.
      env[free_operand] = result_vjp;
    }
    // // only works for GMMs
    // if (forOp.getResultTypes()[0].isa<FloatType>()) {
    //   llvm::errs() << "OUTER FOR OP:\nDifferentiated through " << fv_count
    //                << " free variables:\n";
    // } else {
    //   llvm::errs() << "INNER FOR OP:\nDifferentiated through " << fv_count
    //                << " free variables:\n";
    // }
    // for (auto freeOperand : freeOperands) {
    //   if (isFloatOrFloatTensor(freeOperand.getType())) {
    //     llvm::errs() << "fv: " << freeOperand << "\n";
    //   }
    // }

    return;
  }

  size_t op_index = 0;
  for (Value operand : op->getOperands()) {
    // TODO: This is a bandaid over proper activity analysis
    if (!isFloatOrFloatTensor(operand.getType())) {
      continue;
    }
    // Compute the pullback (VJP).
    // TODO: Gotta be a better way to structure/abstract this. It's
    // essentially a huge switch statement on the operator name.
    if (op->getNumResults() == 0) {
      // op->emitError("op had zero results");
      llvm_unreachable("op had zero results");
      return;
    }
    Value vjp_value = env[op->getResult(0)];
    // We assume the value is dead if it hasn't been come across yet.
    if (!vjp_value) {
      return;
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
    } else if (opName == "arith.negf") {
      vjp_value = rewriter.create<arith::NegFOp>(op->getLoc(), vjp_value);
    } else if (opName == "std.select") {
      // auto selectOp = dyn_cast<mlir::SelectOp>(op);
      llvm_unreachable("std.select not yet implemented");
    } else if (opName == "math.exp") {
      auto expOp = dyn_cast<math::ExpOp>(op);
      vjp_value = rewriter.create<arith::MulFOp>(expOp.getLoc(), vjp_value,
                                                 expOp.getResult());
    } else if (opName == "math.sin") {
      auto cos = rewriter.create<math::CosOp>(op->getLoc(), operand);
      vjp_value = rewriter.create<arith::MulFOp>(op->getLoc(), cos, vjp_value);
    } else if (opName == "math.cos") {
      auto sin = rewriter.create<math::SinOp>(op->getLoc(), operand);
      vjp_value = rewriter.create<arith::MulFOp>(op->getLoc(), sin, vjp_value);
      vjp_value = rewriter.create<arith::NegFOp>(op->getLoc(), vjp_value);
    } else if (opName == "math.tanh") {
      // There's no builtin hyperbolic cos in the math dialect, so we need to
      // express the formula here.
      auto exp = rewriter.create<math::ExpOp>(op->getLoc(), operand);
      auto negexp = rewriter.create<math::ExpOp>(
          op->getLoc(), rewriter.create<arith::NegFOp>(op->getLoc(), operand));
      auto numerator =
          rewriter.create<arith::AddFOp>(op->getLoc(), exp, negexp);
      auto half = constLike(op->getLoc(), operand, 0.5, rewriter);
      auto cosh = rewriter.create<arith::MulFOp>(op->getLoc(), numerator, half);
      auto coshsquared =
          rewriter.create<arith::MulFOp>(op->getLoc(), cosh, cosh);
      vjp_value =
          rewriter.create<arith::DivFOp>(op->getLoc(), vjp_value, coshsquared);
    } else if (opName == "math.log") {
      vjp_value = rewriter.create<arith::DivFOp>(op->getLoc(), vjp_value,
                                                 op->getOperand(0));
    } else if (opName == "math.sqrt") {
      auto half = constLike(op->getLoc(), operand, 0.5, rewriter);
      vjp_value = rewriter.create<arith::MulFOp>(op->getLoc(), vjp_value, half);
      // This is a bit of a math trick. Note the result is sqrt(operand)
      vjp_value = rewriter.create<arith::DivFOp>(op->getLoc(), vjp_value,
                                                 op->getResult(0));
    } else if (opName == "std.call") {
      if (!isFloatOrFloatTensor(operand.getType())) {
        continue;
      }
      vjp_value = reverseCallOp(dyn_cast<CallOp>(op), moduleOp, vjp_value,
                                op_index, rewriter);
    } else if (opName == "tensor.extract") {
      if (op_index > 0) {
        continue;
      }
      auto extractOp = dyn_cast<tensor::ExtractOp>(op);
      bool requires_add = env[operand] != nullptr;
      auto space = requires_add ? env[operand]
                                : getZero(operand.getLoc(), operand, rewriter,
                                          /*init=*/true);
      Value new_val = vjp_value;
      if (requires_add) {
        auto before = rewriter.create<tensor::ExtractOp>(op->getLoc(), space,
                                                         extractOp.indices());
        new_val =
            rewriter.create<arith::AddFOp>(op->getLoc(), before, vjp_value);
      }
      env[operand] = rewriter.create<tensor::InsertOp>(
          op->getLoc(), new_val, space, extractOp.indices());
      continue;
    } else if (opName == "tensor.insert") {
      if (op_index > 0) {
        continue;
      }
      // I don't know if this is super general, but it works for the case we're
      // currently dealing with. Written to support a special case scf.for ops,
      // which is conceptually a linalg.generic with slices.
      auto insertOp = dyn_cast<tensor::InsertOp>(op);
      vjp_value = rewriter.create<tensor::ExtractOp>(op->getLoc(), vjp_value,
                                                     insertOp.indices());

      // The destination is effectively the same memory location as the result.
      env[insertOp.dest()] = env[insertOp.result()];
    } else if (opName == "tensor.extract_slice") {
      auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op);
      auto resultType =
          extractSliceOp.getResult().getType().cast<RankedTensorType>();
      bool requires_add = env[operand] != nullptr;
      auto space = requires_add ? env[operand]
                                : getZero(operand.getLoc(), operand, rewriter,
                                          /*init=*/true);
      Value new_value = vjp_value;
      if (requires_add) {
        auto before = rewriter.create<tensor::ExtractSliceOp>(
            op->getLoc(), resultType, space, extractSliceOp.getMixedOffsets(),
            extractSliceOp.getMixedSizes(), extractSliceOp.getMixedStrides());
        new_value =
            rewriter.create<arith::AddFOp>(op->getLoc(), before, vjp_value);
      }
      env[operand] = rewriter.create<tensor::InsertSliceOp>(
          operand.getLoc(), operand.getType(), new_value, space,
          extractSliceOp.offsets(), extractSliceOp.sizes(),
          extractSliceOp.strides(), extractSliceOp.static_offsets(),
          extractSliceOp.static_sizes(), extractSliceOp.static_strides());
      continue;
    } else if (opName == "tensor.insert_slice") {
      if (op_index > 0) {
        continue;
      }
      auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op);
      vjp_value = rewriter.create<tensor::ExtractSliceOp>(
          op->getLoc(), insertSliceOp.getSourceType(), vjp_value,
          insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
          insertSliceOp.getMixedStrides());
      env[insertSliceOp.dest()] = env[insertSliceOp.getResult()];
    } else if (opName == "linalg.generic") {
      auto genericOp = dyn_cast<linalg::GenericOp>(op);
      if (op_index > static_cast<size_t>(genericOp.getNumInputs() - 1))
        continue;

      // Additionally compute adjoints for all free variables. We only want
      // this to run once, hence the if op_index == 0.
      if (op_index == 0) {
        // Also update the output gradients
        for (auto it :
             llvm::zip(genericOp.getOutputOperands(), genericOp.getResults())) {
          env[std::get<0>(it)->get()] = env[std::get<1>(it)];
        }
        llvm::SmallDenseSet<Value> freeOperands;
        collectFreeVars(genericOp.getBody(), genericOp.getBodyRegion(),
                        freeOperands);
        for (auto freeOperand : freeOperands) {
          if (!isFloatOrFloatTensor(freeOperand.getType())) {
            continue;
          }
          // Not totally sure if we can use the VJP value as-is, watch out
          // for bugs.
          auto out = reverseGenericOp(genericOp, moduleOp, freeOperand,
                                      vjp_value, -1, rewriter);
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
      if (!isFloatOrFloatTensor(operand.getType())) {
        continue;
      }
      vjp_value = reverseGenericOp(genericOp, moduleOp, operand, vjp_value,
                                   op_index, rewriter);
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
    } else if (opName == "linalg.matvec") {
      if (op_index > 1)
        continue;
      if (op_index == 0) {
        auto matvecOp = dyn_cast<linalg::MatvecOp>(op);
        // TODO: This is probably a bandaid solution.
        env[matvecOp.getOutputOperand(0)->get()] = env[matvecOp.getResult(0)];
        // Broadcast the gradient signal
        assert(operand.getType().isa<RankedTensorType>() &&
               "matvec input was not a ranked tensor type");
        SmallVector<AffineMap, 3> indexingMaps(
            op->getNumOperands(), rewriter.getMultiDimIdentityMap(2));
        indexingMaps[0] = indexingMaps[0].getSubMap({0});
        indexingMaps[1] = indexingMaps[1].getSubMap({1});
        auto opType = operand.getType().dyn_cast<RankedTensorType>();
        SmallVector<StringRef, 6> iteratorTypes(opType.getRank(),
                                                getParallelIteratorTypeName());
        auto outerProductOp = rewriter.create<linalg::GenericOp>(
            operand.getLoc(),
            /*resultTensorTypes=*/opType,
            /*inputs=*/ValueRange({vjp_value, op->getOperand(1)}),
            /*outputs=*/ValueRange({operand}), indexingMaps, iteratorTypes,
            /*doc=*/"Vector-vector outer product",
            /*library call=*/"souter",
            [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
              Value mul_res = builder.create<arith::MulFOp>(loc, regionArgs[0],
                                                            regionArgs[1]);
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
            {getReductionIteratorTypeName(), getParallelIteratorTypeName()});

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
              auto mul_res = builder.create<arith::MulFOp>(loc, regionArgs[0],
                                                           regionArgs[1]);
              auto reduced =
                  builder.create<arith::AddFOp>(loc, mul_res, regionArgs[2]);
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
            {getParallelIteratorTypeName(), getReductionIteratorTypeName()});
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
              auto mul_res = builder.create<arith::MulFOp>(loc, regionArgs[0],
                                                           regionArgs[1]);
              auto reduced =
                  builder.create<arith::AddFOp>(loc, mul_res, regionArgs[2]);
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
        SmallVector<StringRef, 6> iteratorTypes(opType.getRank(),
                                                getParallelIteratorTypeName());
        auto outerProductOp = rewriter.create<linalg::GenericOp>(
            operand.getLoc(),
            /*resultTensorTypes=*/opType,
            /*inputs=*/ValueRange({vjp_value, op->getOperand(0)}),
            /*outputs=*/ValueRange({operand}), indexingMaps, iteratorTypes,
            [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
              Value mul_res = builder.create<arith::MulFOp>(loc, regionArgs[0],
                                                            regionArgs[1]);
              builder.create<linalg::YieldOp>(loc, mul_res);
            });
        vjp_value = outerProductOp.getResult(0);
      }
    } else if (opName == "linalg.matmul") {
      if (op_index > 1) {
        continue;
      }
      Value zero = env[operand] ? env[operand]
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
      SmallVector<StringRef, 6> iteratorTypes({getParallelIteratorTypeName(),
                                               getReductionIteratorTypeName(),
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
            Value mul_res = builder.create<arith::MulFOp>(loc, regionArgs[0],
                                                          regionArgs[1]);
            Value add_res =
                builder.create<arith::AddFOp>(loc, regionArgs[2], mul_res);
            builder.create<linalg::YieldOp>(loc, add_res);
          });
      vjp_value = matmulOp.getResult(0);
    } else if (opName == "linalg.batch_matmul") {
      if (op_index > 1) {
        continue;
      }
      vjp_value =
          reverseBatchMatmul(op, operand, vjp_value, op_index, rewriter);
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

Value reverseGenericOp(linalg::GenericOp op, ModuleOp moduleOp, Value operand,
                       Value vjp_value, int op_index,
                       ConversionPatternRewriter &rewriter) {
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
  auto outputShape = output.getType().dyn_cast_or_null<ShapedType>();
  assert(outputShape && outputShape.hasRank() &&
         "output must be a ranked type");
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
            populateVJP(rop, moduleOp, bbEnv, rewriter);
          }
        }

        // This add operation is required in the case of undoing
        // reductions. It might be possible to omit this, if the
        // output argument is never used in the primal, or perhaps if
        // the primal iterator types do not include reductions.
        // I'm not entirely sure how best to check if we can omit this.
        auto new_operand = op_index == -1 ? operand : regionArgs[op_index];
        if (!bbEnv[new_operand]) {
          rewriter.create<linalg::YieldOp>(loc,
                                           getZero(loc, new_operand, rewriter));
        } else if (outputShape.getRank() !=
                   static_cast<int64_t>(numIterators)) {
          Value add_res = rewriter.create<arith::AddFOp>(
              loc, bbEnv[new_operand], regionArgs[regionArgs.size() - 1]);

          rewriter.create<linalg::YieldOp>(loc, add_res);
        } else {
          rewriter.create<linalg::YieldOp>(loc, bbEnv[new_operand]);
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

// Augment the primal for loop to cache iteration values.
void populatePrimalCache(scf::ForOp forOp, ConversionPatternRewriter &rewriter,
                         DenseMap<Value, Value> &val_to_cached) {
  PatternRewriter::InsertionGuard insertionGuard(rewriter);

  // Determine which values to cache.
  SmallVector<Value> valuesToCache;
  for (auto iterOp : forOp.getRegionIterArgs()) {
    if (iterOp.getType().isIntOrIndexOrFloat()) {
      valuesToCache.push_back(iterOp);
    }
  }
  // As a bit of a hack, cache the values using a MemRef because it's easier
  // than modifying the iter arguments to properly use tensors.
  rewriter.setInsertionPoint(forOp);
  auto cacheSize =
      rewriter
          .create<arith::SubIOp>(forOp.getLoc(), forOp.upperBound(),
                                 forOp.lowerBound())
          .getResult();

  SmallVector<Value> caches;
  for (auto cacheVal : valuesToCache) {
    assert(cacheVal.getType().isIntOrIndexOrFloat() &&
           "Caching non-scalars not yet supported");
    auto primalCache = rewriter.create<memref::AllocOp>(
        cacheVal.getLoc(),
        MemRefType::get({/*dynamic size*/ -1}, cacheVal.getType()), cacheSize);
    caches.push_back(primalCache.getResult());
  }

  rewriter.setInsertionPoint(&forOp.getBody()->front());
  for (auto cpair : llvm::zip(caches, valuesToCache)) {
    auto ccache = std::get<0>(cpair);
    auto valToCache = std::get<1>(cpair);
    rewriter.create<memref::StoreOp>(valToCache.getLoc(), valToCache, ccache,
                                     forOp.getInductionVar());
    val_to_cached[valToCache] = ccache;
  }
}

ValueRange reverseForOp(scf::ForOp forOp, ValueRange free_operands,
                        Value vjp_value, size_t result_idx,
                        DenseMap<Value, Value> outer_env,
                        ConversionPatternRewriter &rewriter) {
  PatternRewriter::InsertionGuard insertionGuard(rewriter);
  // Record the ops to clone before augmenting the primal with the caches.
  auto primalOps = forOp.getLoopBody().getOps();
  DenseMap<Value, Value> iterArgsToCached;
  populatePrimalCache(forOp, rewriter, iterArgsToCached);
  SmallVector<Value> operandsWithIV{
      forOp.getInductionVar(),
      // This is only valid under certain conditions, i.e. if the result was
      // only read once.
      forOp.getRegionIterArgs()[result_idx]};
  SmallVector<Value> iterArgsInit({vjp_value});

  for (auto free_operand : free_operands) {
    auto space = outer_env[free_operand]
                     ? outer_env[free_operand]
                     : getZero(free_operand.getLoc(), free_operand, rewriter,
                               /*init=*/true);
    iterArgsInit.push_back(space);
  }
  auto adjointFor = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.lowerBound(), forOp.upperBound(), forOp.step(),
      iterArgsInit,
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
        SmallVector<Value> regionArgs;
        Value idx = builder.create<arith::SubIOp>(loc, forOp.upperBound(), iv);
        Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
        idx = builder.create<arith::SubIOp>(loc, idx, one);
        regionArgs.push_back(idx);
        regionArgs.push_back(forOp.getResult(result_idx));
        for (auto cpair : iterArgsToCached) {
          operandsWithIV.push_back(cpair.first);
          auto loaded = builder.create<memref::LoadOp>(cpair.second.getLoc(),
                                                       cpair.second, idx);
          regionArgs.push_back(loaded);
        }
        auto primalRegionOps = cloneBasicBlock(
            primalOps, builder, /*new=*/regionArgs, /*old=*/operandsWithIV,
            /*offsetInputs=*/false);

        DenseMap<Value, Value> env;
        for (size_t i = 0; i < free_operands.size(); i++) {
          env[free_operands[i]] =
              iterArgs[iterArgs.size() - free_operands.size() + i];
        }
        SmallVector<Value> adjointResults;
        for (auto it = primalRegionOps.rbegin(); it != primalRegionOps.rend();
             it++) {
          auto op = *it;
          auto opName = op->getName().getStringRef();
          if (opName == "scf.yield") {
            Value operand = op->getOperand(result_idx);
            env[operand] = iterArgs[0];
            rewriter.setInsertionPointAfter(op);
            rewriter.eraseOp(op);
          } else {
            auto forModule = forOp->getParentOfType<ModuleOp>();
            assert(forModule && "forModule was null");
            populateVJP(op, forOp->getParentOfType<ModuleOp>(), env, rewriter);
          }
        }

        if (env[forOp.getResult(result_idx)]) {
          adjointResults.push_back(env[forOp.getResult(result_idx)]);
        } else {
          // The primal result was unused in the primal loop body.
          adjointResults.push_back(iterArgs[0]);
        }

        for (size_t i = 0; i < free_operands.size(); i++) {
          auto free_operand = free_operands[i];
          if (!env[free_operand]) {
            // Assume free_operand is not active.
            env[free_operand] =
                getZero(free_operand.getLoc(), free_operand, rewriter);
          }
          adjointResults.push_back(env[free_operand]);
        }
        rewriter.create<scf::YieldOp>(loc, adjointResults);
      });
  return adjointFor.getResults().drop_front(1);
}

Value reverseTensorExtractOp(tensor::ExtractOp op, Value operand,
                             Value vjp_value, OpBuilder &builder) {
  // Using a constant tensor is causing issues here. We need to
  // explicitly allocate a space using init_tensor.
  auto tensorType = op.tensor().getType().dyn_cast<ShapedType>();
  assert(tensorType.hasStaticShape() &&
         "only static shapes are currently supported");
  auto zero = getZero(operand.getLoc(), op.result(), builder);
  Value space = builder.create<linalg::InitTensorOp>(
      operand.getLoc(), ValueRange{}, tensorType.getShape(),
      op.result().getType());
  if (tensorType.getRank() != 0) {
    space = builder.create<linalg::FillOp>(operand.getLoc(), zero, space)
                .getResult(0);
  }
  return builder.create<tensor::InsertOp>(op.getLoc(), vjp_value, space,
                                          op.indices());
}

Value reverseCallOp(CallOp op, ModuleOp moduleOp, Value vjp_value,
                    size_t op_index, ConversionPatternRewriter &rewriter) {
  auto *context = op.getContext();
  std::stringstream gradFuncStream;
  gradFuncStream << "__grad_" << op.callee().str() << "_arg" << op_index;
  auto gradFuncName = gradFuncStream.str();
  assert(moduleOp && "moduleOp was null");
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
