#include "Standalone/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"

using namespace mlir;

namespace {
void runTopDownAnalysis(FuncOp primalFunc, ArrayAttr gradientsOf,
                        llvm::SmallDenseSet<Value> &topDownActive) {
  SmallVector<Value> frontier;
  if (!gradientsOf) {
    frontier.push_back(primalFunc.getArgument(0));
  } else {
    for (auto idxAttr : gradientsOf) {
      auto argIndex = idxAttr.dyn_cast<IntegerAttr>().getValue().getSExtValue();
      frontier.push_back(primalFunc.getArgument(argIndex));
    }
  }

  while (!frontier.empty()) {
    Value val = frontier.pop_back_val();
    if (!topDownActive.contains(val) && !val.getType().isIntOrIndex()) {
      topDownActive.insert(val);
      for (auto user : val.getUsers()) {
        if (auto scfYield = dyn_cast_or_null<scf::YieldOp>(user)) {
          if (auto forOp =
                  dyn_cast_or_null<scf::ForOp>(scfYield->getParentOp())) {
            // Only include directly relevant yielded values.
            for (auto pair :
                 llvm::zip(scfYield.getOperands(), forOp.getResults())) {
              if (std::get<0>(pair) == val) {
                frontier.push_back(std::get<1>(pair));
              }
            }
            frontier.push_back(forOp.getResult(0));
          }
        }
        for (auto result : user->getResults()) {
          frontier.push_back(result);
        }
      }
    }
  }

  // llvm::errs() << "topDownActive:\n";
  // for (auto v : topDownActive) {
  //   llvm::errs() << v << "\n";
  // }
}

void runBottomUpAnalysis(FuncOp primalFunc,
                         llvm::SmallDenseSet<Value> &bottomUpActive) {
  assert(primalFunc.getCallableResults().size() == 1 &&
         "Expected primal to have one result");
  assert(primalFunc.getBody().hasOneBlock() &&
         "Expected body to have one block");
  // get the terminator and traverse bottom-up
  auto *terminator = primalFunc.getBody().getBlocks().front().getTerminator();
  SmallVector<Value> frontier{terminator->getOperands()};
  while (!frontier.empty()) {
    Value val = frontier.pop_back_val();
    if (!bottomUpActive.contains(val) && !val.getType().isIntOrIndex()) {
      bottomUpActive.insert(val);
      if (auto definingOp = val.getDefiningOp()) {
        if (dyn_cast_or_null<arith::ConstantOp>(definingOp)) {
          bottomUpActive.erase(val);
          continue;
        }
        if (auto forOp = dyn_cast_or_null<scf::ForOp>(definingOp)) {
          // Find the index of the used result
          int result_idx = -1;
          for (size_t idx = 0; idx < forOp.getNumResults(); idx++) {
            if (forOp.getResult(idx) == val) {
              result_idx = idx;
              break;
            }
          }
          assert(result_idx != -1 && "Result was not found");
          auto yieldOp = dyn_cast<scf::YieldOp>(
              forOp.getLoopBody().getBlocks().front().getTerminator());
          frontier.push_back(yieldOp.getOperand(result_idx));
        }
        for (auto operand : definingOp->getOperands()) {
          frontier.push_back(operand);
        }
      }
    }
  }

  // llvm::errs() << "bottomUpActive:\n";
  // for (auto v : bottomUpActive) {
  //   llvm::errs() << v << "\n";
  // }
}
} // namespace

void mlir::runActivityAnalysis(LAGradContext &ctx, FuncOp primalFunc,
                               ArrayAttr gradientsOf) {
  llvm::SmallDenseSet<Value> topDownActive;
  llvm::SmallDenseSet<Value> bottomUpActive;
  runTopDownAnalysis(primalFunc, gradientsOf, topDownActive);
  runBottomUpAnalysis(primalFunc, bottomUpActive);

  // Set intersection
  // llvm::errs() << "Active values:\n";
  for (auto td : topDownActive) {
    if (bottomUpActive.contains(td)) {
      ctx.activeValues.insert(td);
      // llvm::errs() << td << "\n";
    }
  }

  // llvm::SmallVector<Value> frontier{terminator->getOperand(0)};
  // while (!frontier.empty()) {
  //   Value val = frontier.pop_back_val();
  //   if (!liveSet.contains(val) && !val.getType().isIntOrIndex()) {
  //     liveSet.insert(val);
  //     auto definingOp = val.getDefiningOp();
  //     if (definingOp) {
  //       if (dyn_cast_or_null<arith::ConstantOp>(definingOp)) {
  //         liveSet.erase(val);
  //         continue;
  //       }
  //       auto ifOp = dyn_cast_or_null<scf::IfOp>(definingOp);
  //       if (ifOp) {
  //         assert(ifOp.getNumResults() == 1 &&
  //                "Expected if op to have 1 result");
  //         runActivityAnalysis(ifOp.thenYield(), {}, liveSet);
  //         runActivityAnalysis(ifOp.elseYield(), {}, liveSet);
  //       }
  //       for (auto operand : definingOp->getOperands())
  //         frontier.push_back(operand);
  //     }
  //   }
  // }
}
