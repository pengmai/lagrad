#include "Standalone/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
using namespace mlir;
static inline void markAsCache(Operation *op) {
  op->setAttr("lagrad_cache", UnitAttr::get(op->getContext()));
}

void populatePrimalCaches(LAGradContext &ctx, FuncOp primalFunc,
                          ConversionPatternRewriter &rewriter) {
  for (auto tbrVal : ctx.toBeRecorded) {
    auto &op =
        *(tbrVal.getDefiningOp() ?: tbrVal.getParentRegion()->getParentOp());
    if (op.getParentOfType<FuncOp>() != primalFunc) {
      continue;
    }
    auto loc = op.getLoc();
    // Allocate caches
    // get the dims of the cache to allocate
    PatternRewriter::InsertionGuard insertionGuard(rewriter);
    SmallVector<Value> upper_bounds;
    SmallVector<Value> lower_bounds;
    SmallVector<Value> induction_vars;
    auto parent = tbrVal.getDefiningOp() ? op.getParentOfType<scf::ForOp>()
                                         : cast<scf::ForOp>(&op);
    // These indices are in reverse order of what you'd expect, going from the
    // inside loop out
    while (parent) {
      induction_vars.push_back(parent.getInductionVar());
      // Assume the step sizes are 1.
      upper_bounds.push_back(parent.upperBound());
      lower_bounds.push_back(parent.lowerBound());
      rewriter.setInsertionPoint(parent);
      parent = parent->getParentOfType<scf::ForOp>();
    }

    SmallVector<Value> dynamicSizes;
    SmallVector<int64_t> shape;
    for (auto tup : llvm::zip(lower_bounds, upper_bounds)) {
      shape.push_back(-1); // Dynamic size
      auto size = rewriter.create<arith::SubIOp>(loc, std::get<1>(tup),
                                                 std::get<0>(tup));
      markAsCache(size);
      dynamicSizes.push_back(size.getResult());
    }
    if (auto rtt = tbrVal.getType().dyn_cast_or_null<RankedTensorType>()) {
      shape.insert(shape.end(), rtt.getShape().begin(), rtt.getShape().end());
      auto cache = rewriter.create<memref::AllocOp>(
          loc, MemRefType::get(shape, rtt.getElementType()), dynamicSizes);
      markAsCache(cache);
      ctx.debug_names[cache] =
          "<primal cache for " + ctx.debug_names[tbrVal] + ">";

      rewriter.setInsertionPointAfterValue(tbrVal);
      // Write to the cache
      auto rctx = rewriter.getContext();
      auto intToAttr = [&](int64_t i) {
        return IntegerAttr::get(IntegerType::get(rctx, 64), i);
      };
      SmallVector<OpFoldResult> mixedOffsets;
      mixedOffsets.reserve(induction_vars.size() + rtt.getRank());
      SmallVector<OpFoldResult> mixedSizes;
      SmallVector<OpFoldResult> mixedStrides;
      for (size_t i = 0; i < induction_vars.size(); i++) {
        mixedOffsets.push_back(induction_vars[i]);
        mixedSizes.push_back(intToAttr(1));
        mixedStrides.push_back(intToAttr(1));
      }
      for (int i = 0; i < rtt.getRank(); i++) {
        mixedOffsets.push_back(intToAttr(0));
        mixedSizes.push_back(intToAttr(rtt.getShape()[i]));
        mixedStrides.push_back(intToAttr(1));
      }
      auto resultType = memref::SubViewOp::inferRankReducedResultType(
                            rtt.getRank(), cache.getType(), mixedOffsets,
                            mixedSizes, mixedStrides)
                            .cast<MemRefType>();
      auto view = rewriter.create<memref::SubViewOp>(
          loc, resultType, cache, mixedOffsets, mixedSizes, mixedStrides);
      markAsCache(view);
      ctx.debug_names[view] =
          "<write view for caching " + ctx.debug_names[tbrVal] + ">";
      auto memref = rewriter.create<memref::BufferCastOp>(
          loc, MemRefType::get(rtt.getShape(), rtt.getElementType()), tbrVal);
      markAsCache(memref);
      ctx.debug_names[memref] =
          "<casted memref for writing " + ctx.debug_names[tbrVal] + ">";
      markAsCache(rewriter.create<linalg::CopyOp>(loc, memref, view));
      ctx.tbrCachedVals[tbrVal] = cache;
    } else {
      auto cache = rewriter.create<memref::AllocOp>(
          loc, MemRefType::get(shape, tbrVal.getType()), dynamicSizes);
      markAsCache(cache);
      ctx.debug_names[cache] =
          "<primal cache for " + ctx.debug_names[tbrVal] + ">";
      rewriter.setInsertionPointAfterValue(tbrVal);

      // Write to the cache
      markAsCache(
          rewriter.create<memref::StoreOp>(loc, tbrVal, cache, induction_vars));
      ctx.tbrCachedVals[tbrVal] = cache;
    }
  }
}
} // namespace mlir
