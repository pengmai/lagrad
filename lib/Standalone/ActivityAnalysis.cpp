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
    if (!topDownActive.contains(val) && isFloatOrFloatTensor(val.getType())) {
      topDownActive.insert(val);
      for (auto user : val.getUsers()) {
        if (auto scfYield = dyn_cast_or_null<scf::YieldOp>(user)) {
          if (auto forOp =
                  dyn_cast_or_null<scf::ForOp>(scfYield->getParentOp())) {
            // Only include directly relevant yielded values.
            // I believe we only need the region iter args and not the iter
            // operand because this is top down analysis.
            for (auto tup :
                 llvm::zip(scfYield.getOperands(), forOp.getResults(),
                           forOp.getRegionIterArgs())) {
              if (std::get<0>(tup) == val) {
                frontier.push_back(std::get<1>(tup));
                frontier.push_back(std::get<2>(tup));
              }
            }
          } else if (auto ifOp =
                         dyn_cast_or_null<scf::IfOp>(scfYield->getParentOp())) {
            for (auto pair :
                 llvm::zip(scfYield.getOperands(), ifOp.getResults())) {
              if (std::get<0>(pair) == val) {
                frontier.push_back(std::get<1>(pair));
              }
            }
          }
        }
        for (auto result : user->getResults()) {
          frontier.push_back(result);
        }
      }
    }
  }
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
    if (!bottomUpActive.contains(val) && isFloatOrFloatTensor(val.getType())) {
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
        } else if (auto ifOp = dyn_cast_or_null<scf::IfOp>(definingOp)) {
          int result_idx = -1;
          for (size_t idx = 0; idx < ifOp.getNumResults(); idx++) {
            if (ifOp.getResult(idx) == val) {
              result_idx = idx;
              break;
            }
          }
          assert(result_idx != -1 && "Result was not found");
          frontier.push_back(ifOp.thenYield().getOperand(result_idx));
          frontier.push_back(ifOp.elseYield().getOperand(result_idx));
        }
        for (auto operand : definingOp->getOperands()) {
          frontier.push_back(operand);
        }
      }
    }
  }
}
} // namespace

void mlir::runActivityAnalysis(LAGradContext &ctx, FuncOp primalFunc,
                               ArrayAttr gradientsOf) {
  llvm::SmallDenseSet<Value> topDownActive;
  llvm::SmallDenseSet<Value> bottomUpActive;
  runTopDownAnalysis(primalFunc, gradientsOf, topDownActive);
  runBottomUpAnalysis(primalFunc, bottomUpActive);

  // Set intersection
  // llvm::errs() << "Top down active values:\n";
  // for (auto td : topDownActive) {
  //   llvm::errs() << "* " << ctx.debug_names[td] << "\n";
  // }
  // llvm::errs() << "\nBottom up active values:\n";
  // for (auto td : bottomUpActive) {
  //   llvm::errs() << "* " << ctx.debug_names[td] << "\n";
  // }
  for (auto td : topDownActive) {
    if (bottomUpActive.contains(td)) {
      ctx.activeValues.insert(td);
      // llvm::errs() << td << "\n";
    }
  }
  llvm::errs() << "active values:\n";
  for (auto v : ctx.activeValues) {
    llvm::errs() << "  " << ctx.debug_names[v] << "\n";
  }
}
