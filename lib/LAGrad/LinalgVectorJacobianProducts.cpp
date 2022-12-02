#include "LAGrad/Analysis.h"
#include "LAGrad/Logger.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
using llvm::errs;

// Trying to find the right way to abstract over dependency analysis, traversing
// def-use chains, etc. This is a really common operation in LAGrad.
void traverseDependencies(Value source, ValueSet &dependents) {
  // TODO: Handle if, loop maybe
  SmallVector<Value> frontier{source};
  while (!frontier.empty()) {
    Value v = frontier.pop_back_val();
    if (!dependents.contains(v)) {
      dependents.insert(v);
      for (Operation *user : v.getUsers()) {
        for (Value result : user->getResults()) {
          frontier.push_back(result);
        }
      }
    }
  }
}

void findRequired(Value dest, ValueSet &required) {
  SmallVector<Value> frontier{dest};
  while (!frontier.empty()) {
    Value v = frontier.pop_back_val();
    if (!required.contains(v)) {
      required.insert(v);
      if (Operation *definingOp = v.getDefiningOp()) {
        frontier.append(definingOp->getOperands().begin(),
                        definingOp->getOperands().end());
      }
    }
  }
}

void runLinalgGenericActivityAnalysis(LAGradContext &ctx, linalg::GenericOp op,
                                      Value operand, int op_index,
                                      ValueSet &activeValues) {
  SmallVector<Value> frontier;
  if (op_index >= 0) {
    frontier.push_back(op.getBlock()->getArgument(op_index));
  } else {
    frontier.push_back(operand);
  }
  // top down activity analysis.
  // Don't want to traverse out of the linalg body.
  ValueSet topDownActive;
  while (!frontier.empty()) {
    Value v = frontier.pop_back_val();
    if (!topDownActive.contains(v)) {
      topDownActive.insert(v);
      for (Operation *user : v.getUsers()) {
        for (Value result : user->getResults()) {
          frontier.push_back(result);
        }
      }
    }
  }

  ValueSet bottomUpActive;
  frontier.clear();
  assert(op.getBlock()->getTerminator()->getNumOperands() == 1 &&
         "Expected linalg.yield to have 1 operand");
  frontier.push_back(op.getBlock()->getTerminator()->getOperand(0));
  while (!frontier.empty()) {
    Value v = frontier.pop_back_val();
    if (!bottomUpActive.contains(v)) {
      bottomUpActive.insert(v);
      if (Operation *definingOp = v.getDefiningOp()) {
        for (Value operand : definingOp->getOperands()) {
          frontier.push_back(operand);
        }
      }
    }
  }
  for (Value v : bottomUpActive) {
    if (topDownActive.contains(v)) {
      activeValues.insert(v);
    }
  }
}

OpOperandVector runLinalgEffectiveUseAnalysis(linalg::GenericOp op,
                                              LAGradContext &ctx, Value operand,
                                              int op_index,
                                              ValueSet &effectivelyUsed) {
  ValueSet activeValues;
  llvm::SmallDenseMap<Value, ValueSet> adjU;
  runLinalgGenericActivityAnalysis(ctx, op, operand, op_index, activeValues);
  populateAdjointUseSets(ctx, activeValues, op.getBodyRegion(), adjU);
  for (Value v : adjU[op.getBlock()->getTerminator()->getOperand(0)]) {
    effectivelyUsed.insert(v);
  }
  if (ctx.debug_names[op.getResult(0)] == "%vx_next") {
    Logger::yellow("Active values");
    printSet(ctx, activeValues);
    Logger::blue("AdjU");
    printSet(ctx, effectivelyUsed);
  }
  // errs() << "for " << ctx.debug_names[op.getResult(0)] << " op index "
  //        << op_index << "\n";
  // for (Value v : effectivelyUsed) {
  //   errs() << "value: " << ctx.debug_names[v] << "\n";
  // }

  // Need to determine which input operands have dependencies to an adjU value.
  OpOperandVector requiredInputs;
  ValueSet dependents;
  for (auto pair : llvm::enumerate(op.getDpsInputOperands())) {
    dependents.clear();
    traverseDependencies(op.getBlock()->getArgument(pair.index()), dependents);
    if (llvm::any_of(effectivelyUsed, [&dependents](Value used) {
          return dependents.contains(used);
        })) {
      requiredInputs.push_back(pair.value());
    }
  }
  return requiredInputs;
}

// version 1
Value reverseGenericOp(linalg::GenericOp op, LAGradContext &ctx, Value operand,
                       Value vjp_value, int op_index, Value output,
                       ConversionPatternRewriter &rewriter) {
  // Need to ensure:
  // if (op_index > (size_t)genericOp.getNumInputs() - 1)
  //   continue;
  auto numIterators = op.getIteratorTypes().size();
  SmallVector<AffineMap, 6> indexing_maps(
      op->getNumOperands() + 1, rewriter.getMultiDimIdentityMap(numIterators));
  SmallVector<utils::IteratorType, 6> iterator_types(
      numIterators, utils::IteratorType::parallel);

  auto outputShape = output.getType().dyn_cast_or_null<ShapedType>();
  assert(outputShape && outputShape.hasRank() &&
         "output must be a ranked type");
  SmallVector<AffineMap> generic_indexing_maps = op.getIndexingMapsArray();
  unsigned int op_count = op.getNumOperands();
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
        // find the dimensions that don't appear in the results
        AffineMap outputMap = generic_indexing_maps[op_index];
        for (size_t idx = 0; idx < outputMap.getNumDims(); idx++) {
          if (!outputMap.isFunctionOfDim(idx)) {
            iterator_types[idx] = utils::IteratorType::reduction;
          }
        }
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
          Value add_res = rewriter.create<arith::AddFOp>(
              loc, bbEnv[new_operand], regionArgs.back());
          rewriter.create<linalg::YieldOp>(loc, add_res);
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
  adjoint->setAttr("adjoint of " + ctx.debug_names[op.getResult(0)],
                   UnitAttr::get(op.getContext()));

  return adjoint.getResult(0);
}

Value reverseGenericOpV2(linalg::GenericOp op, LAGradContext &ctx,
                         Value operand, Value vjp_value, int op_index,
                         Value output, ConversionPatternRewriter &rewriter) {
  // Need to ensure:
  // if (op_index > (size_t)genericOp.getNumInputs() - 1)
  //   continue;
  ValueSet effectivelyUsed;
  OpOperandVector requiredInputs = runLinalgEffectiveUseAnalysis(
      op, ctx, operand, op_index, effectivelyUsed);
  // if (ctx.debug_names[op.getResult(0)] == "%vx_next") {
  //   Logger::blue("Required inputs");
  //   errs() << requiredInputs.size() << " for op index " << op_index << "\n";
  //   for (OpOperand *v : requiredInputs) {
  //     errs() << "value: " << ctx.debug_names[v->get()] << "\n";
  //   }
  // }
  // errs() << "for " << ctx.debug_names[op.getResult(0)] << " op index "
  //        << op_index << "(" << requiredInputs.size() << ")\n";
  // for (OpOperand *v : requiredInputs) {
  //   errs() << "value: " << ctx.debug_names[v->get()] << "\n";
  // }

  SmallVector<AffineMap, 6> indexingMaps;
  indexingMaps.reserve(op->getNumOperands() + 1);
  for (OpOperand *opOp : requiredInputs) {
    indexingMaps.push_back(op.getMatchingIndexingMap(opOp));
  }
  // vjp indexing map
  indexingMaps.push_back(op.getMatchingIndexingMap(op.getDpsInitOperand(0)));
  // output indexing map (with inferred iterator types).
  auto numIterators = op.getIteratorTypes().size();
  SmallVector<utils::IteratorType, 6> iterator_types(
      numIterators, utils::IteratorType::parallel);
  AffineMap outputMap =
      op_index == -1
          // free variables are assumed to be 0d
          ? rewriter.getMultiDimIdentityMap(numIterators).getSubMap({})
          : op.getMatchingIndexingMap(op.getDpsInputOperand(op_index));
  indexingMaps.push_back(outputMap);
  for (size_t idx = 0; idx < outputMap.getNumDims(); idx++) {
    if (!outputMap.isFunctionOfDim(idx)) {
      iterator_types[idx] = utils::IteratorType::reduction;
    }
  }

  auto outputShape = output.getType().dyn_cast_or_null<ShapedType>();
  assert(outputShape && outputShape.hasRank() &&
         "output must be a ranked type");
  SmallVector<Value> inputs{requiredInputs};
  inputs.push_back(vjp_value);

  DenseMap<Value, Value> bbEnv;
  SmallVector<Value> genericOperands;
  for (Value arg : op.getBodyRegion().getArguments()) {
    genericOperands.push_back(arg);
  }

  BlockAndValueMapping map;
  auto adjoint = rewriter.create<linalg::GenericOp>(
      operand.getLoc(), /*resultTensorType=*/outputShape,
      /*inputs=*/inputs, /*outputs=*/ValueRange({output}),
      /*indexing_maps=*/indexingMaps,
      /*iterator_types=*/iterator_types,
      [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
        PatternRewriter::InsertionGuard insertionGuard(rewriter);
        for (auto tup : llvm::zip(requiredInputs, regionArgs)) {
          OpOperand *opOp = std::get<0>(tup);
          Value newRegionArg = std::get<1>(tup);
          map.map(op.getBlock()->getArgument(opOp->getOperandNumber()),
                  newRegionArg);
        }

        // Clone required primal ops
        DenseMap<Value, ValueSet> effectiveDependencies;
        for (Value usedVal : effectivelyUsed) {
          ValueSet bottomUpDeps;
          findRequired(usedVal, bottomUpDeps);
          effectiveDependencies.insert(std::make_pair(usedVal, bottomUpDeps));
        }
        for (auto &primalOp : op.getBlock()->without_terminator()) {
          // need to clone if this result is required to compute any of the
          // effectively used values.
          if (llvm::any_of(effectivelyUsed, [&primalOp, effectiveDependencies](
                                                Value usedVal) {
                return llvm::any_of(
                    primalOp.getResults(),
                    [&effectiveDependencies, usedVal](Value result) {
                      return effectiveDependencies.lookup(usedVal).contains(
                          result);
                    });
              })) {
            Operation *clonedOp = builder.clone(primalOp, map);
            map.map(primalOp.getResults(), clonedOp->getResults());
          }
          // if (llvm::any_of(primalOp.getResults(),
          //                  [&effectiveDependencies](Value result) {
          //                    return effectivelyUsed.contains(result);
          //                  })) {
          //   Operation *clonedOp = builder.clone(primalOp, map);
          //   map.map(primalOp.getResults(), clonedOp->getResults());
          // }
        }

        // Seed from the terminator
        Operation *terminator = op.getBlock()->getTerminator();
        bbEnv[terminator->getOperand(0)] = regionArgs[regionArgs.size() - 2];
        rewriter.restoreInsertionPoint(builder.saveInsertionPoint());

        for (auto &pop : llvm::reverse(op.getBlock()->without_terminator())) {
          if (isa<arith::CmpFOp>(pop)) {
            continue;
          }
          populateVJP(&pop, ctx, bbEnv, rewriter);
        }

        // This add operation is required in the case of undoing
        // reductions. It might be possible to omit this, if the
        // output argument is never used in the primal, or perhaps if
        // the primal iterator types do not include reductions.
        // I'm not entirely sure how best to check if we can omit this.
        auto new_operand =
            op_index == -1 ? operand : op.getBlock()->getArgument(op_index);
        if (!bbEnv[new_operand]) {
          rewriter.create<linalg::YieldOp>(loc,
                                           getZero(loc, new_operand, rewriter));
        } else if (outputShape.getRank() !=
                   static_cast<int64_t>(numIterators)) {
          Value add_res = rewriter.create<arith::AddFOp>(
              loc, bbEnv[new_operand], regionArgs[regionArgs.size() - 1]);

          rewriter.create<linalg::YieldOp>(loc, add_res);
        } else {
          Value add_res = rewriter.create<arith::AddFOp>(
              loc, bbEnv[new_operand], regionArgs.back());
          rewriter.create<linalg::YieldOp>(loc, add_res);
        }
      });

  // Need to switch primal dependencies to recomputed adjoints because
  // populateVJP currently doesn't switch them (it'd have to use the map.)
  for (Operation &bodyOp : adjoint.getBodyRegion().getOps()) {
    for (auto pair : llvm::enumerate(bodyOp.getOperands())) {
      if (map.contains(pair.value())) {
        bodyOp.setOperand(pair.index(), map.lookup(pair.value()));
      }
    }
  }
  // Cleanup needs to happen after the old-to-new mapping
  for (Operation &bodyOp : adjoint.getBodyRegion().getOps()) {
    if (bodyOp.getNumResults() > 0 && bodyOp.use_empty()) {
      rewriter.eraseOp(&bodyOp);
    }
  }
  adjoint->setAttr("adjoint of " + ctx.debug_names[op.getResult(0)],
                   UnitAttr::get(op.getContext()));

  return adjoint.getResult(0);
}
} // namespace mlir
