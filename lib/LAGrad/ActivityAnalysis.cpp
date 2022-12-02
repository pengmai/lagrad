#include "LAGrad/Logger.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <algorithm>
#include <string>
#define VERBOSITY 0

using namespace mlir;
using llvm::errs;

namespace mlir {
void printSet(LAGradContext &ctx, const ValueSet &set, bool pretty) {
  if (pretty) {
    SmallVector<std::string> names;
    for (auto v : set) {
      if (ctx.debug_names.count(v) == 1) {
        names.push_back(ctx.debug_names[v]);
      } else {
        // Hopefully these values aren't important
        // names.push_back("<value not registered>");
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
void runTopDownDFS(SmallVector<Value> &frontier, ValueSet &out) {
  while (!frontier.empty()) {
    Value val = frontier.pop_back_val();
    if (!out.contains(val)) {
      out.insert(val);
      for (auto user : val.getUsers()) {
        if (auto dimOp = dyn_cast<tensor::DimOp>(user)) {
          continue;
        }
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
        } else if (auto callOp = dyn_cast_or_null<func::CallOp>(user)) {
          auto moduleOp = user->getParentOfType<ModuleOp>();
          assert(moduleOp && "moduleOp was null");
          auto callee =
              moduleOp.lookupSymbol<func::FuncOp>(callOp.getCalleeAttr());
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
        if (auto callOp = dyn_cast_or_null<func::CallOp>(definingOp)) {
          auto moduleOp = definingOp->getParentOfType<ModuleOp>();
          assert(moduleOp && "moduleOp was null");
          auto callee =
              moduleOp.lookupSymbol<func::FuncOp>(callOp.getCalleeAttr());
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

void runTopDownAnalysis(LAGradContext &ctx, func::FuncOp primalFunc,
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
  runTopDownDFS(frontier, topDownActive);
}

void runBottomUpAnalysis(func::FuncOp primalFunc, ValueSet &bottomUpActive) {
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

void populateAdjointUseSets(LAGradContext &ctx, ValueSet &activeValues,
                            Region &region,
                            llvm::SmallDenseMap<Value, ValueSet> &adjU) {
  for (auto &op : region.getOps()) {
    if (auto addOp = dyn_cast_or_null<arith::AddFOp>(&op)) {
      ValueSet addAdjU;
      setUnion(addAdjU, adjU[addOp.getLhs()]);
      setUnion(addAdjU, adjU[addOp.getRhs()]);
      adjU[addOp.getResult()] = addAdjU;
    } else if (auto subOp = dyn_cast_or_null<arith::SubFOp>(&op)) {
      ValueSet subAdjU;
      setUnion(subAdjU, adjU[subOp.getLhs()]);
      setUnion(subAdjU, adjU[subOp.getRhs()]);
      adjU[subOp.getResult()] = subAdjU;
    } else if (auto negOp = dyn_cast_or_null<arith::NegFOp>(&op)) {
      ValueSet negAdjU;
      setUnion(negAdjU, adjU[negOp.getOperand()]);
      adjU[negOp.getResult()] = negAdjU;
    } else if (auto mulOp = dyn_cast_or_null<arith::MulFOp>(&op)) {
      ValueSet mulAdjU;
      if (activeValues.contains(mulOp.getLhs())) {
        mulAdjU.insert(mulOp.getRhs());
        setUnion(mulAdjU, adjU[mulOp.getLhs()]);
      }
      if (activeValues.contains(mulOp.getRhs())) {
        mulAdjU.insert(mulOp.getLhs());
        setUnion(mulAdjU, adjU[mulOp.getRhs()]);
      }
      adjU[mulOp.getResult()] = mulAdjU;
    } else if (auto divOp = dyn_cast_or_null<arith::DivFOp>(&op)) {
      ValueSet divAdjU;
      if (activeValues.contains(divOp.getLhs())) {
        divAdjU.insert(divOp.getRhs());
        setUnion(divAdjU, adjU[divOp.getLhs()]);
      }
      if (activeValues.contains(divOp.getRhs())) {
        divAdjU.insert(divOp.getRhs());
        divAdjU.insert(divOp.getLhs());
        setUnion(divAdjU, adjU[divOp.getRhs()]);
      }
      adjU[divOp.getResult()] = divAdjU;
    } else if (auto logOp = dyn_cast_or_null<math::LogOp>(&op)) {
      ValueSet logAdjU;
      logAdjU.insert(logOp.getOperand());
      // This isn't in the TBR paper equations but is required because we're
      // propagating information about previous ops.
      setUnion(logAdjU, adjU[logOp.getOperand()]);
      adjU[logOp.getResult()] = logAdjU;
    } else if (auto expOp = dyn_cast_or_null<math::ExpOp>(&op)) {
      ValueSet expAdjU;
      expAdjU.insert(expOp.getOperand());
      expAdjU.insert(expOp.getResult());
      setUnion(expAdjU, adjU[expOp.getOperand()]);
      adjU[expOp.getResult()] = expAdjU;
    } else if (auto tanhOp = dyn_cast_or_null<math::TanhOp>(&op)) {
      ValueSet tanhAdjU;
      tanhAdjU.insert(tanhOp.getOperand());
      setUnion(tanhAdjU, adjU[tanhOp.getOperand()]);
      adjU[tanhOp.getResult()] = tanhAdjU;
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(&op)) {
      ValueSet selectAdjU;
      selectAdjU.insert(selectOp.getCondition());
      adjU[selectOp.getResult()] = selectAdjU;
    } else if (auto genericOp = dyn_cast_or_null<linalg::GenericOp>(&op)) {
      ValueSet genericAdjU;
      populateAdjointUseSets(ctx, activeValues, genericOp.getBodyRegion(),
                             adjU);
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
      assert(linalgOp.getNumDpsInputs() == 2 &&
             "Expected named linalg op to have 2 inputs");
      auto lhs = linalgOp.getDpsInputOperand(0)->get();
      auto rhs = linalgOp.getDpsInputOperand(1)->get();
      if (activeValues.contains(lhs)) {
        linalgAdjU.insert(rhs);
        setUnion(linalgAdjU, adjU[lhs]);
      }
      if (activeValues.contains(rhs)) {
        linalgAdjU.insert(lhs);
        setUnion(linalgAdjU, adjU[rhs]);
      }
      adjU[linalgOp->getResult(0)] = linalgAdjU;
    } else if (auto extractOp = dyn_cast_or_null<tensor::ExtractOp>(&op)) {
      ValueSet extractAdjU;
      setUnion(extractAdjU, adjU[extractOp.getTensor()]);
      for (auto idx : extractOp.getIndices()) {
        extractAdjU.insert(idx);
      }
      adjU[extractOp.getResult()] = extractAdjU;
    } else if (auto insertOp = dyn_cast_or_null<tensor::InsertOp>(&op)) {
      if (activeValues.contains(insertOp.getScalar())) {
        ValueSet insertAdjU;
        setUnion(insertAdjU, adjU[insertOp.getScalar()]);
        setUnion(insertAdjU, adjU[insertOp.getDest()]);
        for (auto idx : insertOp.getIndices()) {
          insertAdjU.insert(idx);
        }
        adjU[insertOp.getResult()] = insertAdjU;
      }
    } else if (auto extractSliceOp =
                   dyn_cast_or_null<tensor::ExtractSliceOp>(&op)) {
      ValueSet extractSliceAdjU;
      setUnion(extractSliceAdjU, adjU[extractSliceOp.getSource()]);
      for (auto offs : extractSliceOp.getMixedOffsets()) {
        if (auto val = offs.dyn_cast<Value>()) {
          extractSliceAdjU.insert(val);
        }
      }
      adjU[extractSliceOp.getResult()] = extractSliceAdjU;
    } else if (auto insertSliceOp =
                   dyn_cast_or_null<tensor::InsertSliceOp>(&op)) {
      if (activeValues.contains(insertSliceOp.getSource())) {
        ValueSet insertSliceAdjU;
        setUnion(insertSliceAdjU, adjU[insertSliceOp.getSource()]);
        setUnion(insertSliceAdjU, adjU[insertSliceOp.getDest()]);
        for (auto offs : insertSliceOp.getMixedOffsets()) {
          if (auto val = offs.dyn_cast<Value>()) {
            insertSliceAdjU.insert(val);
          }
        }
        adjU[insertSliceOp.getResult()] = insertSliceAdjU;
      }
    } else if (auto ifOp = dyn_cast_or_null<scf::IfOp>(&op)) {
      if (ifOp.getNumResults() > 0) {
        ValueSet ifAdjU;
        populateAdjointUseSets(ctx, activeValues, ifOp.getThenRegion(), adjU);
        populateAdjointUseSets(ctx, activeValues, ifOp.getElseRegion(), adjU);
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
      }
    } else if (auto forOp = dyn_cast_or_null<scf::ForOp>(&op)) {
      ValueSet forAdjU;
      populateAdjointUseSets(ctx, activeValues, forOp.getBodyRegion(), adjU);
      for (auto operand : forOp.getBody()->getTerminator()->getOperands()) {
        setUnion(forAdjU, adjU[operand]);
      }
      for (auto result : forOp.getResults()) {
        adjU[result] = forAdjU;
      }
    } else if (auto callOp = dyn_cast_or_null<func::CallOp>(&op)) {
      ValueSet callAdjU;
      auto &callRegion =
          ctx.moduleOp.lookupSymbol<func::FuncOp>(callOp.getCalleeAttr())
              .getBody();
      populateAdjointUseSets(ctx, activeValues, callRegion, adjU);
      if (callRegion.hasOneBlock()) {
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
}

void runEffectiveUseAnalysis(LAGradContext &ctx, func::FuncOp primalFunc) {
  // adjU maps results to sets of effectively used values
  if (VERBOSITY >= 1) {
    llvm::errs() << "Running TBR Analysis for func " << primalFunc.getName()
                 << "\n";
  }
  llvm::SmallDenseMap<Value, ValueSet> adjU;
  for (auto &op : primalFunc.getCallableRegion()->getOps()) {
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(&op)) {
      if (VERBOSITY >= 1) {
        llvm::errs() << BOLDCYAN << "Running Effective-Use Analysis for loop "
                     << ctx.debug_names[forOp.getResult(0)] << RESET << "\n";
      }
      populateAdjointUseSets(ctx, ctx.activeValues, forOp.getLoopBody(), adjU);
      if (VERBOSITY >= 3) {
        printAllAdjU(ctx, adjU);
      }
      for (auto operand :
           forOp.getLoopBody().front().getTerminator()->getOperands()) {
        setUnion(ctx.effectivelyUsed, adjU[operand]);
      }

      if (VERBOSITY >= 2) {
        llvm::errs() << "adjU:\n";
        printSet(ctx, ctx.effectivelyUsed);
      }
      forOp.walk([&](Operation *childOp) {
        if (childOp->hasAttr("lagrad_should_cache")) {
          ctx.toBeRecorded.insert(childOp->getResult(0));
        }
      });
      // for (auto val : ctx.effectivelyUsed) {
      //   if (auto *defOp = val.getDefiningOp()) {
      //     if (defOp->hasAttr("lagrad_should_cache")) {
      //       ctx.toBeRecorded.insert(val);
      //     }
      //   }
      //   if (auto defOp =
      //   dyn_cast_or_null<arith::AddFOp>(val.getDefiningOp())) {
      //     ctx.toBeRecorded.insert(defOp.getResult());
      //     errs() << BOLDGREEN
      //            << "addf op: " << ctx.debug_names[defOp.getResult()] <<
      //            RESET
      //            << "\n";
      //   }
      // }

      ValueSet derivedFromIterArgs;
      SmallVector<Value> frontier;
      for (auto arg : forOp.getRegionIterArgs()) {
        frontier.push_back(arg);
      }
      runTopDownDFS(frontier, derivedFromIterArgs);
      if (VERBOSITY >= 2) {
        llvm::errs() << "derived from iter args:\n";
        printSet(ctx, derivedFromIterArgs);
      }

      for (auto v : derivedFromIterArgs) {
        if (ctx.effectivelyUsed.contains(v)) {
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

void mlir::runActivityAnalysis(LAGradContext &ctx, func::FuncOp primalFunc,
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
