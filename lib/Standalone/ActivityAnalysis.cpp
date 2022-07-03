#include "Standalone/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#define DEBUG_AA true

using namespace mlir;

namespace {
using ValueSet = llvm::SmallDenseSet<Value>;

void runTopDownDFS(SmallVector<Value> &frontier, ValueSet &out) {
  while (!frontier.empty()) {
    Value val = frontier.pop_back_val();
    if (!out.contains(val) && isFloatOrFloatTensor(val.getType())) {
      out.insert(val);
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
        } else if (auto callOp = dyn_cast_or_null<CallOp>(user)) {
          auto moduleOp = user->getParentOfType<ModuleOp>();
          assert(moduleOp && "moduleOp was null");
          auto callee = moduleOp.lookupSymbol<FuncOp>(callOp.calleeAttr());
          assert(callee && "callee was null");
          // This will overestimate activity in the case that a function
          // argument isn't active in the callee
          for (auto tup :
               llvm::zip(callOp.getArgOperands(), callee.getArguments())) {
            if (std::get<0>(tup) == val) {
              frontier.push_back(std::get<1>(tup));
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

void runBottomUpDFS(SmallVector<Value> &frontier, ValueSet &out) {
  while (!frontier.empty()) {
    Value val = frontier.pop_back_val();
    if (!out.contains(val) && isFloatOrFloatTensor(val.getType())) {
      out.insert(val);
      if (auto definingOp = val.getDefiningOp()) {
        if (dyn_cast_or_null<arith::ConstantOp>(definingOp)) {
          out.erase(val);
          continue;
        }
        if (auto callOp = dyn_cast_or_null<CallOp>(definingOp)) {
          auto moduleOp = definingOp->getParentOfType<ModuleOp>();
          assert(moduleOp && "moduleOp was null");
          auto callee = moduleOp.lookupSymbol<FuncOp>(callOp.calleeAttr());
          assert(callee && "callee was null");
          assert(callee.getBody().hasOneBlock() &&
                 "expected callee to have one block");
          auto terminator = callee.getBody().front().getTerminator();
          for (auto tup :
               llvm::zip(callOp.getResults(), terminator->getOperands())) {
            if (std::get<0>(tup) == val) {
              frontier.push_back(std::get<1>(tup));
              break;
            }
          }
        } else if (auto forOp = dyn_cast_or_null<scf::ForOp>(definingOp)) {
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
        // If this is a call op, this is oversimplified because some operands
        // might not be active based on the contents of the function body.
        for (auto operand : definingOp->getOperands()) {
          frontier.push_back(operand);
        }
      }
    }
  }
}

void runTopDownAnalysis(FuncOp primalFunc, ArrayAttr gradientsOf,
                        ValueSet &topDownActive) {
  SmallVector<Value> frontier;
  if (!gradientsOf) {
    frontier.push_back(primalFunc.getArgument(0));
  } else {
    for (auto idxAttr : gradientsOf) {
      auto argIndex = idxAttr.dyn_cast<IntegerAttr>().getValue().getSExtValue();
      frontier.push_back(primalFunc.getArgument(argIndex));
    }
  }
  runTopDownDFS(frontier, topDownActive);
}

void runBottomUpAnalysis(FuncOp primalFunc, ValueSet &bottomUpActive) {
  assert(primalFunc.getCallableResults().size() == 1 &&
         "Expected primal to have one result");
  assert(primalFunc.getBody().hasOneBlock() &&
         "Expected body to have one block");
  // get the terminator and traverse bottom-up
  auto *terminator = primalFunc.getBody().getBlocks().front().getTerminator();
  SmallVector<Value> frontier{terminator->getOperands()};
  runBottomUpDFS(frontier, bottomUpActive);
}

static inline void setUnion(ValueSet &a, const ValueSet &b) {
  for (auto v : b) {
    a.insert(v);
  }
}

static inline void printSet(LAGradContext &ctx, const ValueSet &set,
                            bool pretty = true) {
  llvm::errs() << "{";
  size_t i = 0;
  for (auto v : set) {
    if (pretty) {
      llvm::errs() << ctx.debug_names[v];
    } else {
      llvm::errs() << v;
    }
    if (i != set.size() - 1) {
      llvm::errs() << ", ";
    }
    i++;
  }
  llvm::errs() << "}\n";
}

// void runDepsForIterArgs(LAGradContext &ctx)
void populateAdjointUseSets(LAGradContext &ctx, Region &region,
                            llvm::SmallDenseMap<Value, ValueSet> &adjU) {
  for (auto &op : region.getOps()) {
    if (auto addOp = dyn_cast_or_null<arith::AddFOp>(&op)) {
      ValueSet addAdjU;
      setUnion(addAdjU, adjU[addOp.lhs()]);
      setUnion(addAdjU, adjU[addOp.rhs()]);
      adjU[addOp.getResult()] = addAdjU;
    } else if (auto subOp = dyn_cast_or_null<arith::SubFOp>(&op)) {
      ValueSet subAdjU;
      setUnion(subAdjU, adjU[subOp.lhs()]);
      setUnion(subAdjU, adjU[subOp.rhs()]);
      adjU[subOp.getResult()] = subAdjU;
    } else if (auto negOp = dyn_cast_or_null<arith::NegFOp>(&op)) {
      ValueSet negAdjU;
      setUnion(negAdjU, adjU[negOp.operand()]);
      adjU[negOp.getResult()] = negAdjU;
    } else if (auto mulOp = dyn_cast_or_null<arith::MulFOp>(&op)) {
      ValueSet mulAdjU;
      // if (ctx.activeValues.contains(mulOp.lhs())) {
      mulAdjU.insert(mulOp.rhs());
      setUnion(mulAdjU, adjU[mulOp.lhs()]);
      // }
      // if (ctx.activeValues.contains(mulOp.rhs())) {
      mulAdjU.insert(mulOp.lhs());
      setUnion(mulAdjU, adjU[mulOp.rhs()]);
      // }
      adjU[mulOp.getResult()] = mulAdjU;
    } else if (auto divOp = dyn_cast_or_null<arith::DivFOp>(&op)) {
      ValueSet divAdjU;
      // if (ctx.activeValues.contains(divOp.lhs())) {
      divAdjU.insert(divOp.rhs());
      setUnion(divAdjU, adjU[divOp.lhs()]);
      // }
      // if (ctx.activeValues.contains(divOp.rhs())) {
      divAdjU.insert(divOp.lhs());
      setUnion(divAdjU, adjU[divOp.rhs()]);
      // }
      adjU[divOp.getResult()] = divAdjU;
    } else if (auto logOp = dyn_cast_or_null<math::LogOp>(&op)) {
      ValueSet logAdjU;
      logAdjU.insert(logOp.getOperand());
      adjU[logOp.getResult()] = logAdjU;
    } else if (auto expOp = dyn_cast_or_null<math::ExpOp>(&op)) {
      ValueSet expAdjU;
      expAdjU.insert(expOp.getOperand());
      adjU[expOp.getResult()] = expAdjU;
    } else if (auto tanhOp = dyn_cast_or_null<math::TanhOp>(&op)) {
      ValueSet tanhAdjU;
      tanhAdjU.insert(tanhOp.getOperand());
      adjU[tanhOp.getResult()] = tanhAdjU;
    } else if (auto genericOp = dyn_cast_or_null<linalg::GenericOp>(&op)) {
      ValueSet genericAdjU;
      populateAdjointUseSets(ctx, genericOp.getBodyRegion(), adjU);
      for (auto yieldOperand :
           genericOp.getBody()->getTerminator()->getOperands()) {
        llvm::errs() << "frontier init: ";
        printSet(ctx, adjU[yieldOperand], false);
        SmallVector<Value> frontier{adjU[yieldOperand].begin(),
                                    adjU[yieldOperand].end()};
        ValueSet yieldDeps;
        runBottomUpDFS(frontier, yieldDeps);
        llvm::errs() << "yield deps: ";
        printSet(ctx, yieldDeps, false);
        for (auto tup : llvm::zip(genericOp.getOperands(),
                                  genericOp.getBlock()->getArguments())) {
          if (yieldDeps.contains(std::get<1>(tup))) {
            genericAdjU.insert(std::get<0>(tup));
            break;
          }
        }
      }
      assert(genericOp.getNumResults() == 1 &&
             "expected generic op to have one result");
      adjU[genericOp.getResult(0)] = genericAdjU;
    } else if (auto extractSliceOp =
                   dyn_cast_or_null<tensor::ExtractSliceOp>(&op)) {
      ValueSet extractSliceAdjU;
      setUnion(extractSliceAdjU, adjU[extractSliceOp.source()]);
      adjU[extractSliceOp.getResult()] = extractSliceAdjU;
    } else if (auto insertSliceOp =
                   dyn_cast_or_null<tensor::InsertSliceOp>(&op)) {
      if (ctx.activeValues.contains(insertSliceOp.source())) {
        ValueSet insertSliceAdjU;
        setUnion(insertSliceAdjU, adjU[insertSliceOp.source()]);
        setUnion(insertSliceAdjU, adjU[insertSliceOp.dest()]);
        adjU[insertSliceOp.getResult()] = insertSliceAdjU;
      }
    } else if (auto ifOp = dyn_cast_or_null<scf::IfOp>(&op)) {
      ValueSet ifAdjU;
      populateAdjointUseSets(ctx, ifOp.thenRegion(), adjU);
      populateAdjointUseSets(ctx, ifOp.elseRegion(), adjU);
      for (auto operand : ifOp.thenBlock()->getTerminator()->getOperands()) {
        setUnion(ifAdjU, adjU[operand]);
      }
      for (auto operand : ifOp.elseBlock()->getTerminator()->getOperands()) {
        setUnion(ifAdjU, adjU[operand]);
      }
      // This may be a little coarse-grained, assuming adjU flow for multiple
      // arguments when that may not be the case.
      for (auto result : ifOp.getResults()) {
        adjU[result] = ifAdjU;
      }
    } else if (auto callOp = dyn_cast_or_null<CallOp>(&op)) {
      ValueSet callAdjU;
      auto &callRegion =
          ctx.moduleOp.lookupSymbol<FuncOp>(callOp.calleeAttr()).getBody();
      populateAdjointUseSets(ctx, callRegion, adjU);
      assert(callRegion.hasOneBlock() && "Expected function to have one block");
      for (auto returnOperand :
           callRegion.back().getTerminator()->getOperands()) {
        SmallVector<Value> frontier{adjU[returnOperand].begin(),
                                    adjU[returnOperand].end()};
        ValueSet returnDeps;
        runBottomUpDFS(frontier, returnDeps);
        llvm::errs() << "adjU of return operand: ";
        printSet(ctx, adjU[returnOperand]);
        for (auto tup :
             llvm::zip(callRegion.getArguments(), callOp.getOperands())) {
          if (returnDeps.contains(std::get<0>(tup))) {
            callAdjU.insert(std::get<1>(tup));
          }
        }
      }
      llvm::errs() << "call adjointU: ";
      printSet(ctx, callAdjU);
      for (auto result : callOp.getResults()) {
        adjU[result] = callAdjU;
      }
    }
  }
}

void runEffectiveUseAnalysis(LAGradContext &ctx, FuncOp primalFunc) {
  // adjU maps results to sets of effectively used values
  llvm::SmallDenseMap<Value, ValueSet> adjU;
  primalFunc.walk([&](scf::ForOp forOp) {
    ValueSet finalAdjU;
    populateAdjointUseSets(ctx, forOp.getLoopBody(), adjU);
    for (auto operand :
         forOp.getLoopBody().front().getTerminator()->getOperands()) {
      setUnion(finalAdjU, adjU[operand]);
    }

    if (DEBUG_AA) {
      llvm::errs() << "adjU:\n";
      printSet(ctx, finalAdjU);
    }

    ValueSet derivedFromIterArgs;
    SmallVector<Value> frontier;
    for (auto arg : forOp.getRegionIterArgs()) {
      if (ctx.activeValues.contains(arg)) {
        frontier.push_back(arg);
      }
    }
    runTopDownDFS(frontier, derivedFromIterArgs);
    if (DEBUG_AA) {
      llvm::errs() << "derived from iter args:\n";
      printSet(ctx, derivedFromIterArgs);
      // for (auto v : derivedFromIterArgs) {
      //   llvm::errs() << "* " << ctx.debug_names[v] << "\n";
      // }
    }

    ValueSet tbr;
    for (auto v : derivedFromIterArgs) {
      if (finalAdjU.contains(v)) {
        tbr.insert(v);
      }
    }
    if (DEBUG_AA) {
      llvm::errs() << "Final TBR values:\n";
      printSet(ctx, tbr);
    }
  });
}
} // namespace

void mlir::runActivityAnalysis(LAGradContext &ctx, FuncOp primalFunc,
                               ArrayAttr gradientsOf) {
  llvm::SmallDenseSet<Value> topDownActive;
  llvm::SmallDenseSet<Value> bottomUpActive;
  runTopDownAnalysis(primalFunc, gradientsOf, topDownActive);
  runBottomUpAnalysis(primalFunc, bottomUpActive);

  // llvm::errs() << "Top down active values:\n";
  // for (auto td : topDownActive) {
  //   llvm::errs() << "* " << ctx.debug_names[td] << "\n";
  // }
  // llvm::errs() << "\nBottom up active values:\n";
  // for (auto td : bottomUpActive) {
  //   llvm::errs() << "* " << ctx.debug_names[td] << "\n";
  // }

  // Set intersection
  for (auto td : topDownActive) {
    if (bottomUpActive.contains(td)) {
      ctx.activeValues.insert(td);
    }
  }
  if (DEBUG_AA) {
    llvm::errs() << "active values:\n";
    printSet(ctx, ctx.activeValues);
  }
  runEffectiveUseAnalysis(ctx, primalFunc);
}
