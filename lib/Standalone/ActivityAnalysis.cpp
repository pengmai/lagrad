#include "Standalone/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"

using namespace mlir;

void mlir::runActivityAnalysis(Operation *terminator, ValueRange args,
                               llvm::SmallDenseSet<Value> &liveSet) {
  assert(terminator->getNumOperands() == 1 &&
         "Expected terminator to have 1 operand");
  llvm::SmallVector<Value> frontier{terminator->getOperand(0)};
  while (!frontier.empty()) {
    Value val = frontier.pop_back_val();
    if (!liveSet.contains(val) && !val.getType().isIntOrIndex()) {
      liveSet.insert(val);
      auto definingOp = val.getDefiningOp();
      if (definingOp) {
        if (dyn_cast_or_null<arith::ConstantOp>(definingOp)) {
          liveSet.erase(val);
          continue;
        }
        auto ifOp = dyn_cast_or_null<scf::IfOp>(definingOp);
        if (ifOp) {
          assert(ifOp.getNumResults() == 1 &&
                 "Expected if op to have 1 result");
          runActivityAnalysis(ifOp.thenYield(), {}, liveSet);
          runActivityAnalysis(ifOp.elseYield(), {}, liveSet);
        }
        for (auto operand : definingOp->getOperands())
          frontier.push_back(operand);
      }
    }
  }
}
