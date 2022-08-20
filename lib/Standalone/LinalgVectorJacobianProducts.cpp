#include "Standalone/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
Value reverseGenericOp(linalg::GenericOp op, LAGradContext &ctx, Value operand,
                       Value vjp_value, int op_index, Value output,
                       ConversionPatternRewriter &rewriter) {
  // Need to ensure:
  // if (op_index > (size_t)genericOp.getNumInputs() - 1)
  //   continue;
  auto numIterators = op.iterator_types().size();
  SmallVector<AffineMap, 6> indexing_maps(
      op->getNumOperands() + 1, rewriter.getMultiDimIdentityMap(numIterators));
  SmallVector<StringRef, 6> iterator_types(numIterators,
                                           getParallelIteratorTypeName());

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
} // namespace mlir
