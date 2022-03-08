/**
 * The goal of this transformation is to move allocations out of scf.for loops
 * even when they are yielded in the iteration arguments.
 */

#include "Standalone/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
class HoistLoopAllocs : public OpRewritePattern<scf::ForOp> {
public:
  HoistLoopAllocs(MLIRContext *ctx)
      : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::none_of(forOp.getIterOperands(), [](Value iter_op) {
          assert(iter_op && "iter op was null");
          return iter_op.getType().isa<MemRefType>();
        })) {
      return failure();
    }
    auto yieldOp =
        cast<scf::YieldOp>(forOp.getRegion().front().getTerminator());
    SmallVector<Operation *, 4> allocsToHoist;
    forOp.walk([&](memref::AllocOp allocOp) {
      for (auto operand : yieldOp.getOperands()) {
        if (allocOp.getResult() == operand) {
          allocsToHoist.push_back(allocOp);
        }
      }
    });
    if (allocsToHoist.empty()) {
      return failure();
    }
    rewriter.updateRootInPlace(forOp, [&]() {
      if (failed(forOp.moveOutOfLoop(allocsToHoist))) {
        assert(false && "failed to move allocs out of loop");
      }
    });

    // forOp.walk()
    return success();
  }
};
} // namespace

namespace {
struct StandaloneLoopHoistingPass
    : public PassWrapper<StandaloneLoopHoistingPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  StringRef getArgument() const override { return "standalone-loop-hoisting"; }
  StringRef getDescription() const override {
    return "Hoist buffers that are present in loops.";
  }
  void runOnOperation() final {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<HoistLoopAllocs>(patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::Standalone::createLoopHoistingPass() {
  return std::make_unique<StandaloneLoopHoistingPass>();
}
