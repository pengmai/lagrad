#include "Standalone/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
using namespace mlir;
void populatePrimalCaches(LAGradContext &ctx, FuncOp primalFunc,
                          ConversionPatternRewriter &rewriter) {
  for (auto tbrVal : ctx.toBeRecorded) {
    auto &op =
        *(tbrVal.getDefiningOp() ?: tbrVal.getParentRegion()->getParentOp());
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
      dynamicSizes.push_back(
          rewriter
              .create<arith::SubIOp>(loc, std::get<1>(tup), std::get<0>(tup))
              .getResult());
    }
    if (auto rtt = tbrVal.getType().dyn_cast_or_null<RankedTensorType>()) {
      shape.insert(shape.end(), rtt.getShape().begin(), rtt.getShape().end());
      auto cache = rewriter.create<memref::AllocOp>(
          loc, MemRefType::get(shape, rtt.getElementType()), dynamicSizes);
      ctx.debug_names[cache] =
          "<primal cache for " + ctx.debug_names[tbrVal] + ">";

      rewriter.setInsertionPointAfterValue(tbrVal);
      // Write to the cache
      auto slice_layout = getRankReduceSubviewLayout(rtt.getRank(), rewriter);
      auto resultType =
          MemRefType::get(rtt.getShape(), rtt.getElementType(), slice_layout);
      auto rctx = rewriter.getContext();
      auto intToAttr = [&](int64_t i) {
        return IntegerAttr::get(IntegerType::get(rctx, 64), i);
      };
      SmallVector<Attribute> staticOffsets;
      SmallVector<Attribute> staticSizes;
      SmallVector<Attribute> staticStrides;
      for (size_t i = 0; i < induction_vars.size(); i++) {
        staticOffsets.push_back(
            IntegerAttr::get(IntegerType::get(rctx, 64),
                             -9223372036854775808ULL)); // dynamic size
        staticSizes.push_back(intToAttr(1));
        staticStrides.push_back(intToAttr(1));
      }
      for (int i = 0; i < rtt.getRank(); i++) {
        staticOffsets.push_back(intToAttr(0));
        staticSizes.push_back(intToAttr(rtt.getShape()[i]));
        staticStrides.push_back(intToAttr(1));
      }
      auto staticOffset = ArrayAttr::get(rctx, staticOffsets);
      auto staticSize = ArrayAttr::get(rctx, staticSizes);
      auto staticStride = ArrayAttr::get(rctx, staticStrides);
      auto view = rewriter.create<memref::SubViewOp>(
          loc, resultType, cache,
          /*dynamic shapes=*/induction_vars, ValueRange(), ValueRange(),
          /*staticShapes=*/staticOffset, staticSize, staticStride);
      ctx.debug_names[view] =
          "<write view for caching " + ctx.debug_names[tbrVal] + ">";
      auto memref = rewriter.create<memref::BufferCastOp>(
          loc, MemRefType::get(rtt.getShape(), rtt.getElementType()), tbrVal);
      ctx.debug_names[memref] =
          "<casted memref for writing " + ctx.debug_names[tbrVal] + ">";
      rewriter.create<linalg::CopyOp>(loc, memref, view);
      ctx.tbrCachedVals[tbrVal] = cache;
    }
  }
}
} // namespace mlir
