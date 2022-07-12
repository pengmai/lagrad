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

AffineMap getRankReduceSubviewLayout(int64_t resultRank,
                                     ConversionPatternRewriter &rewriter) {
  // if (resultRank == 1) {
  //   return AffineMap::get(
  //       1, 1, rewriter.getAffineDimExpr(0) +
  //       rewriter.getAffineSymbolExpr(0));
  // }
  AffineExpr expr;
  for (int64_t exprIdx = resultRank - 1; exprIdx >= 0; exprIdx--) {
    if (exprIdx == resultRank - 1) {
      expr = rewriter.getAffineDimExpr(exprIdx) *
                 rewriter.getAffineSymbolExpr(resultRank) +
             rewriter.getAffineSymbolExpr(exprIdx);
    } else {
      expr = rewriter.getAffineDimExpr(exprIdx) *
                 rewriter.getAffineSymbolExpr(exprIdx) +
             expr;
    }
  }
  return AffineMap::get(resultRank, resultRank + 1, expr);
}

SmallVector<mlir::Operation *>
cloneBasicBlock(llvm::iterator_range<Region::OpIterator> bbOps,
                OpBuilder &builder, ValueRange regionArgs,
                SmallVector<Value> bbOperands, bool offsetInputs,
                LAGradContext *ctx) {
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
    if (ctx) {
      for (auto tup : llvm::zip(op.getResults(), clonedOp->getResults())) {
        ctx->debug_names[std::get<1>(tup)] =
            "<cloned " + ctx->debug_names[std::get<0>(tup)] + ">";
        if (ctx->activeValues.contains(std::get<0>(tup))) {
          ctx->activeValues.insert(std::get<1>(tup));
        }
      }
    }
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

FuncOp differentiateFunction(FuncOp funcOp, LAGradContext &ctx,
                             ArrayAttr gradientsOf,
                             ConversionPatternRewriter &rewriter,
                             bool topLevel = false) {
  Region *region = funcOp.getCallableRegion();
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
      populateVJP(op, ctx, env, rewriter);
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

Value getZero(Location loc, Value operand, OpBuilder &rewriter, bool init) {
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
  llvm::errs() << "getZero for type " << operand.getType()
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

void collectFreeVarsImpl(SmallVector<Block *> &parentBlocks, ValueSet &out) {
  assert(!parentBlocks.empty() && "parentBlocks was empty");
  for (auto &op : parentBlocks.back()->getOperations()) {
    for (auto operand : op.getOperands()) {
      if (!isFloatOrFloatTensor(operand.getType())) {
        continue;
      }
      auto definingOp = operand.getDefiningOp();
      if (dyn_cast_or_null<arith::ConstantOp>(definingOp)) {
        continue;
      }
      if (llvm::none_of(parentBlocks, [&](Block *parentBlock) {
            return operand.getParentBlock() == parentBlock;
          })) {
        out.insert(operand);
      }
    }
    for (auto &childRegion : op.getRegions()) {
      for (auto &childBlock : childRegion.getBlocks()) {
        SmallVector<Block *> newBlocks{parentBlocks};
        newBlocks.push_back(&childBlock);
        collectFreeVarsImpl(newBlocks, out);
      }
    }
  }
}

// TODO: delete the region argument from here
void collectFreeVars(Block *parentBlock, Region &region, ValueSet &out) {
  SmallVector<Block *> parentBlocks{parentBlock};
  collectFreeVarsImpl(parentBlocks, out);
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

Value addInPlace(Value source, Value dest, OpBuilder &builder) {
  if (source.getType().isa<FloatType>()) {
    return builder.create<arith::AddFOp>(dest.getLoc(), source, dest);
  }
  assert(dest.getType().isa<RankedTensorType>() &&
         "add in place expects a ranked tensor destination");
  auto outputShape = dest.getType().dyn_cast<RankedTensorType>();
  auto rank = outputShape.getRank();
  SmallVector<AffineMap, 2> indexingMaps(2,
                                         builder.getMultiDimIdentityMap(rank));
  SmallVector<StringRef, 1> iteratorTypes(rank, getParallelIteratorTypeName());
  auto addOp = builder.create<linalg::GenericOp>(
      dest.getLoc(),
      /*resultTensorType=*/outputShape,
      /*inputs=*/source, /*outputs=*/dest,
      /*indexing_maps=*/indexingMaps,
      /*iterator_types=*/iteratorTypes,
      /*doc=*/"Add in place",
      /*library call=*/"",
      [&](OpBuilder &bodyBuilder, Location loc, ValueRange regionArgs) {
        auto added = bodyBuilder.create<arith::AddFOp>(loc, regionArgs[0],
                                                       regionArgs[1]);
        bodyBuilder.create<linalg::YieldOp>(loc, added.getResult());
      });
  return addOp.getResult(0);
}

void populateVJP(Operation *op, LAGradContext &ctx,
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
  if (opName == "memref.store") {
    // Store ops are unsupported, so we ignore them for now.
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
      auto result =
          reverseIfOp(ifOp, ctx, freeOperand, vjp_value, env, rewriter);
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
    ValueSet freeOperands;
    collectFreeVars(forOp.getBody(), forOp.getLoopBody(), freeOperands);

    auto free_operand_vec = llvm::to_vector<4>(freeOperands);
    reverseForOp(forOp, ctx, free_operand_vec, vjp_value, result_idx, env,
                 rewriter);
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

  for (size_t op_index = 0; op_index < op->getNumOperands(); op_index++) {
    Value operand = op->getOperand(op_index);
    // TODO: This is a bandaid over proper activity analysis
    if (!isFloatOrFloatTensor(operand.getType())) {
      continue;
    }
    // Compute the pullback (VJP).
    // TODO: Gotta be a better way to structure/abstract this. It's
    // essentially a huge switch statement on the operator name.
    if (op->getNumResults() == 0) {
      // op->emitError("op had zero results");
      op->emitWarning() << "op had zero results";
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
      auto selectOp = dyn_cast<mlir::SelectOp>(op);
      // Assume that the gradient is zero for the branch not taken.
      // Not sure if this is true in general.
      if (op_index == 1) {
        // true branch
        auto zero = getZero(op->getLoc(), operand, rewriter);
        vjp_value = rewriter.create<SelectOp>(
            op->getLoc(), selectOp.condition(), vjp_value, zero);
      } else if (op_index == 2) {
        // false branch
        auto zero = getZero(op->getLoc(), operand, rewriter);
        vjp_value = rewriter.create<SelectOp>(
            op->getLoc(), selectOp.condition(), zero, vjp_value);
      }
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
      vjp_value =
          rewriter.create<arith::DivFOp>(op->getLoc(), vjp_value, operand);
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
      vjp_value = reverseCallOp(dyn_cast<CallOp>(op), ctx, vjp_value, op_index,
                                rewriter);
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
      auto zero = getZero(insertOp.getLoc(), insertOp.scalar(), rewriter);
      auto new_dresult = rewriter.create<tensor::InsertOp>(
          insertOp.getLoc(), zero, env[insertOp.result()], insertOp.indices());
      env[insertOp.dest()] = new_dresult;
      env[insertOp.result()] = new_dresult;
      // env[insertOp.dest()] = env[insertOp.result()];
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
        assert(env[operand].getType() == operand.getType() &&
               "operand and its gradient had different types");
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
      auto zero = getZero(op->getLoc(), insertSliceOp.source(), rewriter);
      auto destGrad = rewriter.create<tensor::InsertSliceOp>(
          op->getLoc(), zero, env[insertSliceOp.getResult()],
          insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
          insertSliceOp.getMixedStrides());
      env[insertSliceOp.dest()] = destGrad;

      // env[insertSliceOp.dest()] = env[insertSliceOp.getResult()];
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
          auto out = reverseGenericOp(genericOp, ctx, freeOperand, vjp_value,
                                      -1, rewriter);
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
      vjp_value = reverseGenericOp(genericOp, ctx, operand, vjp_value, op_index,
                                   rewriter);
    } else if (opName == "linalg.dot") {
      if (op_index > 1)
        continue;
      if (op_index > 1 || !ctx.activeValues.contains(operand))
        continue;

      SmallVector<AffineMap, 6> indexing_maps(
          op->getNumOperands(), rewriter.getMultiDimIdentityMap(1));
      indexing_maps[0] = indexing_maps[0].getSubMap({});
      indexing_maps[1] = indexing_maps[1].getSubMap({0});
      indexing_maps[2] = indexing_maps[2].getSubMap({0});
      auto library_call =
          op_index == 0 ? "sdot_grad_first" : "sdot_grad_second";
      auto output = getZero(operand.getLoc(), operand, rewriter);
      auto adjoint = rewriter.create<linalg::GenericOp>(
          operand.getLoc(), /*resultTensorTypes=*/operand.getType(),
          /*inputs=*/
          ValueRange({vjp_value, op->getOperand(1 - op_index)}),
          /*outputs=*/ValueRange({output}), indexing_maps,
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
      env[operand] = addInPlace(vjp_value, env[operand], rewriter);
    }
  }
}

Value reverseGenericOp(linalg::GenericOp op, LAGradContext &ctx, Value operand,
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

  Operation *yieldOp = nullptr;

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
            // rop->erase();
            rewriter.eraseOp(rop);
            yieldOp = rop;
          } else if (rop->getName().getStringRef() == "arith.cmpf") {
            continue;
          } else {
            populateVJP(rop, ctx, bbEnv, rewriter);
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

  for (auto &bodyOp : adjoint.getBodyRegion().getOps()) {
    // Ugly, but necessary to do some form of cleanup here because non-active
    // primal ops might no longer be in scope if the generic ops are inside a
    // loop. scf.if ops inside the generic body cause this to segfault for some
    // reason.
    for (auto *user : bodyOp.getUsers()) {
      if (bodyOp.getNumResults() > 0 && bodyOp.hasOneUse() && user == yieldOp) {
        rewriter.eraseOp(&bodyOp);
      }
    }
  }

  return adjoint.getResult(0);
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

Value reverseCallOp(CallOp op, LAGradContext &ctx, Value vjp_value,
                    size_t op_index, ConversionPatternRewriter &rewriter) {
  auto *context = op.getContext();
  std::stringstream gradFuncStream;
  gradFuncStream << "__grad_" << op.callee().str() << "_arg" << op_index;
  auto gradFuncName = gradFuncStream.str();
  assert(ctx.moduleOp && "moduleOp was null");
  auto dFuncOp =
      dyn_cast_or_null<FuncOp>(ctx.moduleOp.lookupSymbol(gradFuncName));
  if (!dFuncOp) {
    auto primalFunc =
        dyn_cast<FuncOp>(ctx.moduleOp.lookupSymbol(op.calleeAttr()));
    dFuncOp = copyFunctionDeclaration(primalFunc, gradFuncName, rewriter);

    auto innerGradsOf = ArrayAttr::get(
        context, {IntegerAttr::get(IntegerType::get(context, 64), op_index)});
    runActivityAnalysis(ctx, dFuncOp, innerGradsOf);
    populatePrimalCaches(ctx, dFuncOp, rewriter);
    dFuncOp = differentiateFunction(dFuncOp, ctx, innerGradsOf, rewriter);
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
