#include "Standalone/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include <algorithm>
#include <string>
#define VERBOSITY 0

using namespace mlir;

namespace mlir {
void runTopDownDFS(LAGradContext &ctx, SmallVector<Value> &frontier,
                   ValueSet &out) {
  while (!frontier.empty()) {
    Value val = frontier.pop_back_val();
    if (!out.contains(val) && isFloatOrFloatTensor(val.getType())) {
      out.insert(val);
      for (auto user : val.getUsers()) {
        if (auto genericOp = dyn_cast_or_null<linalg::GenericOp>(user)) {
          for (auto tup : llvm::zip(genericOp.getOperands(),
                                    genericOp.getBodyRegion().getArguments())) {
            if (std::get<0>(tup) == val) {
              frontier.push_back(std::get<1>(tup));
            }
          }
        } else if (auto linalgYieldOp =
                       dyn_cast_or_null<linalg::YieldOp>(user)) {
          auto genericOp =
              cast<linalg::GenericOp>(linalgYieldOp->getParentOp());
          for (auto tup :
               llvm::zip(linalgYieldOp.getOperands(), genericOp.getResults())) {
            if (std::get<0>(tup) == val) {
              frontier.push_back(std::get<1>(tup));
            }
          }
        } else if (auto forOp = dyn_cast_or_null<scf::ForOp>(user)) {
          // Handles where the use is an iter arg operand
          for (auto &operand : forOp.getIterOpOperands()) {
            if (operand.get() == val) {
              frontier.push_back(forOp.getRegionIterArgForOpOperand(operand));
            }
          }
        } else if (auto scfYield = dyn_cast_or_null<scf::YieldOp>(user)) {
          if (auto forOp =
                  dyn_cast_or_null<scf::ForOp>(scfYield->getParentOp())) {
            // Handles where the use is a free variable in a for body.
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
    // if (!out.contains(val) && isFloatOrFloatTensor(val.getType())) {
    if (!out.contains(val)) {
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
        } else if (auto genericOp =
                       dyn_cast_or_null<linalg::GenericOp>(definingOp)) {
          assert(genericOp.getNumResults() == 1);
          frontier.push_back(
              genericOp.getBody()->getTerminator()->getOperand(0));
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

void runTopDownAnalysis(LAGradContext &ctx, FuncOp primalFunc,
                        ArrayAttr gradientsOf, ValueSet &topDownActive) {
  SmallVector<Value> frontier;
  if (!gradientsOf) {
    frontier.push_back(primalFunc.getArgument(0));
  } else {
    for (auto idxAttr : gradientsOf) {
      auto argIndex = idxAttr.dyn_cast<IntegerAttr>().getValue().getSExtValue();
      frontier.push_back(primalFunc.getArgument(argIndex));
    }
  }
  runTopDownDFS(ctx, frontier, topDownActive);
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
  if (pretty) {
    SmallVector<std::string> names;
    for (auto v : set) {
      if (ctx.debug_names.count(v) == 1) {
        names.push_back(ctx.debug_names[v]);
      } else {
        names.push_back("<value not registered>");
      }
    }
    std::sort(names.begin(), names.end());
    llvm::errs() << "{";
    size_t i = 0;
    for (auto name : names) {
      llvm::errs() << name;
      if (i != names.size() - 1) {
        llvm::errs() << ", ";
      }
      i++;
    }
    llvm::errs() << "}\n";
  }
}

static inline void printAllAdjU(LAGradContext &ctx,
                                llvm::SmallDenseMap<Value, ValueSet> &adjU) {
  llvm::errs() << "*** PRINTING ALL ADJU SETS***\n";
  SmallVector<Value> names;
  for (auto pair : adjU) {
    names.push_back(pair.first);
  }
  std::sort(names.begin(), names.end(),
            [&](const Value &a, const Value &b) -> bool {
              return ctx.debug_names[a] < ctx.debug_names[b];
            });
  for (auto name : names) {
    llvm::errs() << ctx.debug_names[name] << ": ";
    printSet(ctx, adjU[name]);
  }
  llvm::errs() << "*** DONE ***\n";
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
      if (ctx.activeValues.contains(mulOp.lhs())) {
        mulAdjU.insert(mulOp.rhs());
        setUnion(mulAdjU, adjU[mulOp.lhs()]);
      }
      if (ctx.activeValues.contains(mulOp.rhs())) {
        mulAdjU.insert(mulOp.lhs());
        setUnion(mulAdjU, adjU[mulOp.rhs()]);
      }
      adjU[mulOp.getResult()] = mulAdjU;
    } else if (auto divOp = dyn_cast_or_null<arith::DivFOp>(&op)) {
      ValueSet divAdjU;
      if (ctx.activeValues.contains(divOp.lhs())) {
        divAdjU.insert(divOp.rhs());
        setUnion(divAdjU, adjU[divOp.lhs()]);
      }
      if (ctx.activeValues.contains(divOp.rhs())) {
        divAdjU.insert(divOp.lhs());
        setUnion(divAdjU, adjU[divOp.rhs()]);
      }
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
        // Add free operands that are in adjU
        for (auto v : adjU[yieldOperand]) {
          if (v.getParentBlock() != genericOp.getBlock()) {
            genericAdjU.insert(v);
          }
        }

        SmallVector<Value> frontier{adjU[yieldOperand].begin(),
                                    adjU[yieldOperand].end()};
        ValueSet yieldDeps;
        runBottomUpDFS(frontier, yieldDeps);
        for (auto tup : llvm::zip(genericOp.getOperands(),
                                  genericOp.getBlock()->getArguments())) {
          if (yieldDeps.contains(std::get<1>(tup))) {
            genericAdjU.insert(std::get<0>(tup));
            setUnion(genericAdjU, adjU[std::get<0>(tup)]);
          }
        }
      }
      assert(genericOp.getNumResults() == 1 &&
             "expected generic op to have one result");
      adjU[genericOp.getResult(0)] = genericAdjU;
    } else if (op.getName().getStringRef() == "linalg.dot" ||
               op.getName().getStringRef() == "linalg.matmul" ||
               op.getName().getStringRef() == "linalg.matvec" ||
               op.getName().getStringRef() == "linalg.vecmat" ||
               op.getName().getStringRef() == "linalg.batch_matmul") {
      ValueSet linalgAdjU;
      auto linalgOp = cast<linalg::LinalgOp>(&op);
      assert(linalgOp.getNumInputs() == 2 &&
             "Expected named linalg op to have 2 inputs");
      auto lhs = linalgOp.getInputOperand(0)->get();
      auto rhs = linalgOp.getInputOperand(1)->get();
      if (ctx.activeValues.contains(lhs)) {
        linalgAdjU.insert(rhs);
        setUnion(linalgAdjU, adjU[lhs]);
      }
      if (ctx.activeValues.contains(rhs)) {
        linalgAdjU.insert(lhs);
        setUnion(linalgAdjU, adjU[rhs]);
      }
      adjU[linalgOp->getResult(0)] = linalgAdjU;
    } else if (auto extractOp = dyn_cast_or_null<tensor::ExtractOp>(&op)) {
      ValueSet extractAdjU;
      setUnion(extractAdjU, adjU[extractOp.tensor()]);
      adjU[extractOp.getResult()] = extractAdjU;
    } else if (auto insertOp = dyn_cast_or_null<tensor::InsertOp>(&op)) {
      if (ctx.activeValues.contains(insertOp.scalar())) {
        ValueSet insertAdjU;
        setUnion(insertAdjU, adjU[insertOp.scalar()]);
        setUnion(insertAdjU, adjU[insertOp.dest()]);
        adjU[insertOp.getResult()] = insertAdjU;
      }
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
    } else if (auto forOp = dyn_cast_or_null<scf::ForOp>(&op)) {
      ValueSet forAdjU;
      populateAdjointUseSets(ctx, forOp.getBodyRegion(), adjU);
      for (auto operand : forOp.getBody()->getTerminator()->getOperands()) {
        setUnion(forAdjU, adjU[operand]);
      }
      for (auto result : forOp.getResults()) {
        adjU[result] = forAdjU;
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
        for (auto tup :
             llvm::zip(callRegion.getArguments(), callOp.getOperands())) {
          if (returnDeps.contains(std::get<0>(tup))) {
            callAdjU.insert(std::get<1>(tup));
            setUnion(callAdjU, adjU[std::get<1>(tup)]);
          }
        }
      }
      for (auto result : callOp.getResults()) {
        adjU[result] = callAdjU;
      }
    }
  }
}

void runEffectiveUseAnalysis(LAGradContext &ctx, FuncOp primalFunc) {
  // adjU maps results to sets of effectively used values
  llvm::SmallDenseMap<Value, ValueSet> adjU;
  for (auto &op : primalFunc.getCallableRegion()->getOps()) {
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(&op)) {
      ValueSet finalAdjU;
      populateAdjointUseSets(ctx, forOp.getLoopBody(), adjU);
      if (VERBOSITY >= 3) {
        printAllAdjU(ctx, adjU);
      }
      for (auto operand :
           forOp.getLoopBody().front().getTerminator()->getOperands()) {
        setUnion(finalAdjU, adjU[operand]);
      }

      if (VERBOSITY >= 2) {
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
      runTopDownDFS(ctx, frontier, derivedFromIterArgs);
      if (VERBOSITY >= 2) {
        llvm::errs() << "derived from iter args:\n";
        printSet(ctx, derivedFromIterArgs);
      }

      for (auto v : derivedFromIterArgs) {
        if (finalAdjU.contains(v)) {
          ctx.toBeRecorded.insert(v);
        }
      }
      if (VERBOSITY >= 1) {
        llvm::errs() << "Final TBR values (" << ctx.toBeRecorded.size()
                     << "):\n";
        printSet(ctx, ctx.toBeRecorded);
      }
    }
  }
}
} // namespace mlir

void mlir::runActivityAnalysis(LAGradContext &ctx, FuncOp primalFunc,
                               ArrayAttr gradientsOf) {
  llvm::SmallDenseSet<Value> topDownActive;
  llvm::SmallDenseSet<Value> bottomUpActive;
  runTopDownAnalysis(ctx, primalFunc, gradientsOf, topDownActive);
  runBottomUpAnalysis(primalFunc, bottomUpActive);

  if (VERBOSITY >= 3) {
    llvm::errs() << "Top down active values: ";
    printSet(ctx, topDownActive);
    llvm::errs() << "\nBottom up active values: ";
    printSet(ctx, bottomUpActive);
  }

  // Set intersection
  for (auto td : topDownActive) {
    if (bottomUpActive.contains(td)) {
      ctx.activeValues.insert(td);
    }
  }
  if (VERBOSITY >= 2) {
    llvm::errs() << "active values for function " << primalFunc.getName()
                 << " (" << ctx.activeValues.size() << "):\n";
    printSet(ctx, ctx.activeValues);
  }
  runEffectiveUseAnalysis(ctx, primalFunc);
}
