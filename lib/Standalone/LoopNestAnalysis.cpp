#include "Standalone/Analysis.h"

namespace mlir {
// Find the induction variable of a loop that may be iterating in reverse.
Value getInductionVar(scf::ForOp op, LoopNest &loopNest) {
  if (op.getInductionVar().hasOneUse()) {
    if (auto subiOp =
            dyn_cast<arith::SubIOp>(*op.getInductionVar().getUsers().begin())) {
      // case 1: (ub - iv) - 1
      if (subiOp.lhs() == op.upperBound() &&
          subiOp.rhs() == op.getInductionVar() &&
          subiOp.getResult().hasOneUse()) {
        if (auto secondSubiOp = dyn_cast<arith::SubIOp>(
                *subiOp.getResult().getUsers().begin())) {
          auto rhs = dyn_cast_or_null<arith::ConstantIndexOp>(
              secondSubiOp.rhs().getDefiningOp());
          if (secondSubiOp.lhs() == subiOp.getResult() && rhs &&
              rhs.value() == 1) {
            loopNest.ivComputation.insert(subiOp);
            loopNest.ivComputation.insert(secondSubiOp);
            return secondSubiOp.getResult();
          }
        }
      } else {
        // case 2: (const ub - 1) - iv
        auto constLHS = dyn_cast_or_null<arith::ConstantIndexOp>(
            subiOp.lhs().getDefiningOp());
        auto constUpperBound = dyn_cast_or_null<arith::ConstantIndexOp>(
            op.upperBound().getDefiningOp());
        if (subiOp.rhs() == op.getInductionVar() && constLHS &&
            constUpperBound &&
            constLHS.value() == constUpperBound.value() - 1) {
          loopNest.ivComputation.insert(subiOp);
          return subiOp.getResult();
        }
      }
    }
  }
  return op.getInductionVar();
}

using llvm::errs;
Optional<std::pair<Value, Value>> traverseTiedLoopOperands(Value regionArg) {
  // errs() << "traversing tied loop operands for region arg " << regionArg
  //        << "\n";
  Value tensor = regionArg;
  Value result;
  while (auto parentForOp =
             dyn_cast<scf::ForOp>(tensor.getParentRegion()->getParentOp())) {
    OpOperand *iterOperand =
        llvm::find_if(parentForOp.getIterOpOperands(), [&](OpOperand &operand) {
          return parentForOp.getRegionIterArgForOpOperand(operand) == tensor;
        });
    if (iterOperand == parentForOp.getIterOpOperands().end()) {
      return llvm::None;
    }
    tensor = iterOperand->get();
    result = parentForOp.getResultForOpOperand(*iterOperand);
  }
  return std::make_pair(tensor, result);
}

bool isPaired(tensor::ExtractOp read, tensor::InsertOp write) {
  if (!(read.tensor() == write.dest() && read.indices() == write.indices())) {
    return false;
  }

  // The read tensor should only be used by the write.
  DominanceInfo dom;
  for (Operation *user : read.tensor().getUsers()) {
    if (dom.properlyDominates(read.getResult(), user) && user != write) {
      return false;
    }
  }
  return true;
}

Optional<LoopNest> parseLoopNest(scf::ForOp op) {
  if (op->getParentOfType<scf::ForOp>()) {
    return llvm::None;
  }

  LoopNest loopNest;
  WalkResult result = op->walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
    auto childLoops = forOp.getBody()->getOps<scf::ForOp>();
    loopNest.loops.push_back(forOp);
    loopNest.inductionVars.push_back(getInductionVar(forOp, loopNest));
    if (childLoops.empty()) {
      // this is the innermost loop
      if (!llvm::all_of(
              forOp.getBody()->getOperations(), [](Operation &childOp) {
                return isa<tensor::ExtractOp>(childOp) ||
                       isa<tensor::InsertOp>(childOp) ||
                       llvm::all_of(childOp.getResultTypes(), [](Type type) {
                         return type.isIntOrIndexOrFloat();
                       });
              })) {
        return WalkResult::interrupt();
      }
      // All yielded operands should be the result of insert ops
      DenseSet<Operation *> activeReads;
      for (Value yieldOperand :
           forOp.getBody()->getTerminator()->getOperands()) {
        auto insertOp =
            dyn_cast_or_null<tensor::InsertOp>(yieldOperand.getDefiningOp());
        if (!insertOp) {
          return WalkResult::interrupt();
        }
        // need to find all the reads that affect the scalar within the loop
        // body
        SmallVector<Value> frontier{insertOp.scalar()};
        while (!frontier.empty()) {
          Value val = frontier.pop_back_val();
          if (Operation *definingOp = val.getDefiningOp()) {
            if (auto extractOp = dyn_cast<tensor::ExtractOp>(definingOp)) {
              if (!activeReads.contains(definingOp)) {
                activeReads.insert(definingOp);
              }
            } else {
              for (auto operand : definingOp->getOperands()) {
                frontier.push_back(operand);
              }
            }
          }
        }

        for (auto read : activeReads) {
          auto extractOp = cast<tensor::ExtractOp>(read);
          if (extractOp.tensor() == insertOp.dest()) {
            // This extract op is an output operand
            tensor::InsertOp tmpInsert = insertOp;
            unsigned numWrites = 1;
            while (auto prevWrite = dyn_cast_or_null<tensor::InsertOp>(
                       tmpInsert.dest().getDefiningOp())) {
              tmpInsert = prevWrite;
              numWrites++;
            }

            loopNest.outputRegionArgs.push_back(tmpInsert.dest());
            loopNest.outputPerIterWrites.push_back(numWrites);
            auto maybeOutputOperand =
                traverseTiedLoopOperands(tmpInsert.dest());
            if (!maybeOutputOperand.hasValue()) {
              return WalkResult::interrupt();
            }
            loopNest.outputTensorOperands.push_back(
                maybeOutputOperand.getValue().first);
            loopNest.results.push_back(maybeOutputOperand.getValue().second);
          } else {
            SmallVector<AffineExpr, 4> resultExprs;
            for (Value idxVal : extractOp.indices()) {
              ptrdiff_t idx = std::distance(
                  loopNest.inductionVars.begin(),
                  std::find(loopNest.inductionVars.begin(),
                            loopNest.inductionVars.end(), idxVal));
              if (idx < static_cast<long>(loopNest.inductionVars.size())) {
                resultExprs.push_back(getAffineDimExpr(idx, op.getContext()));
              } else if (auto constIndexOp =
                             dyn_cast_or_null<arith::ConstantIndexOp>(
                                 idxVal.getDefiningOp())) {
                resultExprs.push_back(getAffineConstantExpr(
                    constIndexOp.value(), op.getContext()));
              } else {
                // The read indices were not a function of the induction vars,
                // nor were they constants.
                // The value -1 here is used as a placeholder for "unknown".
                resultExprs.push_back(
                    getAffineConstantExpr(-1, op.getContext()));
                // return WalkResult::interrupt();
              }
            }
            loopNest.inputRegionArgs.push_back(extractOp.tensor());
            loopNest.inputMaps.push_back(AffineMap::get(
                loopNest.inductionVars.size(),
                /*symbolCount=*/0, resultExprs, op.getContext()));
            auto maybeInputOperand =
                traverseTiedLoopOperands(extractOp.tensor());
            // errs() << "looking at input operand: " << extractOp.tensor()
            //        << "\n";
            if (!maybeInputOperand.hasValue()) {
              // extractOp.emitWarning() << "interrupt 4";
              return WalkResult::interrupt();
            }
            loopNest.inputTensorOperands.push_back(
                maybeInputOperand.getValue().first);
          }
        }
      }
    } else {
      // Ensure we only have one loop directly inside this one.
      int numLoops = std::distance(childLoops.begin(), childLoops.end());
      if (numLoops > 1 ||
          !llvm::all_of(
              forOp.getBody()->getOperations(), [](Operation &childOp) {
                return isa<scf::ForOp>(childOp) ||
                       llvm::all_of(childOp.getResultTypes(), [](Type type) {
                         // This is a placeholder to test that the bodies of
                         // middle loops only contain the index reversal
                         // computation.
                         return type.isIntOrIndex();
                       });
              })) {
        // errs() << "interrupt 6\n";
        return WalkResult::interrupt();
      }
      // Ensure the yield operands of this loop match the results of the child
      // loop.
      scf::ForOp childLoop = *childLoops.begin();
      if (!llvm::all_of_zip(forOp.getBody()->getTerminator()->getOperands(),
                            childLoop.getResults(),
                            [](Value yieldOperand, Value childResult) {
                              return yieldOperand == childResult;
                            })) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return llvm::None;
  }
  // errs() << "successfully parsed loop nest\n";
  // for (auto input : loopNest.inputTensorOperands) {
  //   errs() << "input: " << input << "\n";
  // }
  return loopNest;
}

LoopNestAnalysis::LoopNestAnalysis(Operation *op) {
  op->walk([&](scf::ForOp forOp) {
    auto maybeNest = parseLoopNest(forOp);
    if (maybeNest.hasValue()) {
      forOpMapping[forOp] = maybeNest.getValue();
    }
  });
}

Optional<LoopNest> LoopNestAnalysis::getLoopNest(scf::ForOp op) const {
  if (forOpMapping.count(op) == 0) {
    return llvm::None;
  }
  return forOpMapping.lookup(op);
}

bool LoopNestAnalysis::isLoopNest(scf::ForOp op) const {
  return forOpMapping.count(op) != 0;
}

} // namespace mlir
