#include "LAGrad/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

namespace mlir {
using namespace mlir;
using llvm::errs;
void DEPRECATEDpopulatePrimalCache(
    scf::ForOp forOp, ConversionPatternRewriter &rewriter,
    SmallVector<std::pair<Value, Value>> &val_to_cached) {
  PatternRewriter::InsertionGuard insertionGuard(rewriter);
  SmallVector<Value> valuesToCache;
  for (auto iterOp : forOp.getRegionIterArgs()) {
    if (iterOp.getType().isIntOrIndexOrFloat()) {
      valuesToCache.push_back(iterOp);
    }
    // We naïvely cache all 1d tensor values here.
    if (auto rankedType =
            iterOp.getType().dyn_cast_or_null<RankedTensorType>()) {
      // Commenting this line out assumes the for op represents a scan and thus
      // doesn't need any tensor caching to access its intermediate values.

      // For LSTMs, we're representing the state as a 3d tensor.
      // if (rankedType.getRank() == 1 || rankedType.getRank() == 3) {
      // valuesToCache.push_back(iterOp);
      // }
    }
  }
  // As a bit of a hack, cache the values using a MemRef because it's easier
  // than modifying the iter arguments to properly use tensors.
  rewriter.setInsertionPoint(forOp);
  auto cacheSize =
      rewriter
          .create<arith::SubIOp>(forOp.getLoc(), forOp.getUpperBound(),
                                 forOp.getLowerBound())
          .getResult();

  SmallVector<Value> caches;
  for (auto cacheVal : valuesToCache) {
    if (auto rankedType =
            cacheVal.getType().dyn_cast_or_null<RankedTensorType>()) {
      // fully cache every 1d iter arg.
      SmallVector<int64_t> shape;
      shape.reserve(rankedType.getRank() + 1);
      shape.push_back(-1); // dynamic size
      for (auto size : rankedType.getShape()) {
        shape.push_back(size);
      }
      auto primalCache = rewriter.create<memref::AllocOp>(
          cacheVal.getLoc(),
          MemRefType::get(shape, rankedType.getElementType()), cacheSize);
      caches.push_back(primalCache.getResult());
    } else {
      // Allocate space for storing scalars.
      auto primalCache = rewriter.create<memref::AllocOp>(
          cacheVal.getLoc(),
          MemRefType::get({/*dynamic size*/ -1}, cacheVal.getType()),
          cacheSize);
      caches.push_back(primalCache.getResult());
    }
  }

  rewriter.setInsertionPoint(&forOp.getBody()->front());
  for (auto cpair : llvm::zip(caches, valuesToCache)) {
    auto ccache = std::get<0>(cpair);
    auto valToCache = std::get<1>(cpair);
    if (auto tensorType =
            valToCache.getType().dyn_cast_or_null<RankedTensorType>()) {
      auto slice_layout =
          getRankReduceSubviewLayout(tensorType.getRank(), rewriter);
      auto resultType = MemRefType::get(
          tensorType.getShape(), tensorType.getElementType(), slice_layout);
      auto ctx = rewriter.getContext();
      auto intToAttr = [&](int64_t i) {
        return IntegerAttr::get(IntegerType::get(ctx, 64), i);
      };
      SmallVector<Attribute> staticOffsets{
          IntegerAttr::get(IntegerType::get(ctx, 64), -9223372036854775808ULL)};
      SmallVector<Attribute> staticSizes{intToAttr(1)};
      SmallVector<Attribute> staticStrides{intToAttr(1)};
      for (int i = 0; i < tensorType.getRank(); i++) {
        staticOffsets.push_back(intToAttr(0));
        staticSizes.push_back(intToAttr(tensorType.getShape()[i]));
        staticStrides.push_back(intToAttr(1));
      }
      auto staticOffset = ArrayAttr::get(ctx, staticOffsets);
      auto staticSize = ArrayAttr::get(ctx, staticSizes);
      auto staticStride = ArrayAttr::get(ctx, staticStrides);
      auto view = rewriter.create<memref::SubViewOp>(
          valToCache.getLoc(), resultType, ccache,
          /*dynamic shapes=*/ValueRange(forOp.getInductionVar()), ValueRange(),
          ValueRange(),
          /*staticShapes=*/staticOffset, staticSize, staticStride);
      auto memref = rewriter.create<bufferization::ToMemrefOp>(
          valToCache.getLoc(),
          MemRefType::get(tensorType.getShape(), tensorType.getElementType()),
          valToCache);
      rewriter.create<memref::CopyOp>(valToCache.getLoc(), memref, view);
    } else {
      rewriter.create<memref::StoreOp>(valToCache.getLoc(), valToCache, ccache,
                                       forOp.getInductionVar());
    }
    val_to_cached.push_back(std::pair<Value, Value>(valToCache, ccache));
  }
}

void reverseForOpV2(scf::ForOp forOp, LAGradContext &ctx,
                    ValueRange free_operands, Value vjp_value,
                    size_t result_idx, DenseMap<Value, Value> &outer_env,
                    ConversionPatternRewriter &rewriter) {
  PatternRewriter::InsertionGuard insertionGuard(rewriter);
  // Record the ops to clone before augmenting the primal with the caches.
  auto primalOps = forOp.getLoopBody().getOps();
  SmallVector<Value> operandsWithIV{
      forOp.getInductionVar(),
      // This is only valid under certain conditions, i.e. if the result was
      // only read once.
      forOp.getRegionIterArgs()[result_idx]};
  // TODO: This would be cleaner if the iter_args order was preserved, and free
  // operands were added after.
  // it goes [result value, ...free_operands, ...primal_iter_args]
  SmallVector<Value> iterArgsInit({vjp_value});
  // By construction, free operands come before iter arg grads, which is a
  // little awkward.
  SmallVector<Value> inputOperands{free_operands};
  inputOperands.reserve(free_operands.size() + forOp.getNumIterOperands());
  for (size_t i = 0; i < forOp.getNumIterOperands(); i++) {
    auto iterOperand = forOp.getIterOperands()[i];
    if (isFloatOrFloatTensor(iterOperand.getType()) && i != result_idx) {
      inputOperands.push_back(iterOperand);
    }
  }

  for (auto input_operand : inputOperands) {
    // Allocate spaces for the gradients of each input operand, if required.
    auto space = outer_env[input_operand]
                     ? outer_env[input_operand]
                     : getZero(input_operand.getLoc(), input_operand, rewriter,
                               /*init=*/true);
    if (auto *op = space.getDefiningOp()) {
      op->setAttr("gradient space for " + ctx.debug_names[input_operand],
                  UnitAttr::get(op->getContext()));
    }
    iterArgsInit.push_back(space);
  }

  DenseMap<Value, Value> oldToCloned;
  auto adjointFor = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      iterArgsInit,
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
        Value idx = builder.create<arith::SubIOp>(loc, forOp.getUpperBound(), iv);
        Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
        idx = builder.create<arith::AddIOp>(loc, idx, forOp.getLowerBound());
        idx = builder.create<arith::SubIOp>(loc, idx, one);
        SmallVector<Value> regionArgs;
        regionArgs.push_back(idx);
        regionArgs.push_back(forOp.getResult(result_idx));
        SmallVector<Value> replacedPrimalIterArgs;
        Value vjp_op = nullptr;

        oldToCloned[forOp.getInductionVar()] = idx;
        // Emit reads to the cached values
        auto markAsCache = [&](Operation *op, std::string name) {
          op->setAttr("cached " + name, UnitAttr::get(op->getContext()));
        };
        for (auto tbrPair : ctx.tbrCachedVals) {
          auto tbrVal = tbrPair.first;
          auto cache = tbrPair.second;
          // If tbrVal doesn't have a defining op, assume it's a region arg of
          // an scf.for op.
          auto parent =
              tbrVal.getDefiningOp()
                  ? tbrVal.getDefiningOp()->getParentOfType<scf::ForOp>()
                  : cast<scf::ForOp>(tbrVal.getParentRegion()->getParentOp());
          if (parent == forOp) {
            SmallVector<Value> induction_vars;
            auto canAvoidCaching = [&](scf::ForOp parent) {
              return isLoopParallel(parent) &&
                     llvm::none_of(parent.getResults(), [&](Value val) {
                       return ctx.effectivelyUsed.contains(val);
                     });
            };
            while (parent && !canAvoidCaching(parent)) {
              induction_vars.push_back(parent.getInductionVar());
              parent = parent->getParentOfType<scf::ForOp>();
            }
            if (auto rtt =
                    tbrVal.getType().dyn_cast_or_null<RankedTensorType>()) {
              auto rctx = rewriter.getContext();
              auto intToAttr = [&](int64_t i) {
                return IntegerAttr::get(IntegerType::get(rctx, 64), i);
              };
              SmallVector<OpFoldResult> mixedOffsets;
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
              auto resultType =
                  memref::SubViewOp::inferRankReducedResultType(
                      rtt.getRank(), cache.getType().cast<MemRefType>(),
                      mixedOffsets, mixedSizes, mixedStrides)
                      .cast<MemRefType>();
              auto view = builder.create<memref::SubViewOp>(
                  loc, resultType, cache, mixedOffsets, mixedSizes,
                  mixedStrides);
              markAsCache(view, ctx.debug_names[tbrVal]);
              ctx.debug_names[view] =
                  "<read view for caching " + ctx.debug_names[tbrVal] + ">";
              auto casted = builder.create<memref::CastOp>(
                  loc, view.getResult(),
                  MemRefType::get(rtt.getShape(), rtt.getElementType()));
              markAsCache(casted, ctx.debug_names[tbrVal]);
              ctx.debug_names[casted] =
                  "<casted memref for reading " + ctx.debug_names[tbrVal] + ">";
              auto loaded = builder.create<bufferization::ToTensorOp>(
                  loc, casted.getResult());
              markAsCache(loaded, ctx.debug_names[tbrVal]);
              ctx.debug_names[loaded] =
                  "<loaded tensor from cached " + ctx.debug_names[tbrVal] + ">";
              oldToCloned[tbrVal] = loaded;
            } else {
              auto loaded =
                  builder.create<memref::LoadOp>(loc, cache, induction_vars);
              markAsCache(loaded, ctx.debug_names[tbrVal]);
              ctx.debug_names[loaded] =
                  "<loaded scalar from cached " + ctx.debug_names[tbrVal] + ">";
              oldToCloned[tbrVal] = loaded;
            }
          }
        }

        // Clone ops that don't depend on the region iter args
        for (auto &pop : primalOps) {
          if (!pop.hasAttr("lagrad_cache") && pop.getNumResults() > 0) {
            SmallVector<Value> frontier{pop.getResults()};
            ValueSet deps;
            runBottomUpDFS(frontier, deps);
            auto inDependSet = [&](Value v) { return deps.contains(v); };
            if (llvm::none_of(
                    pop.getResults(),
                    [&](Value v) { return ctx.toBeRecorded.contains(v); }) &&
                llvm::none_of(forOp.getRegionIterArgs(), inDependSet)) {
              auto cloned = builder.clone(pop);
              // Yet another bandaid: this allows the primal outer loop in GMM
              // to be totally erased because it's unused.
              if (isa<scf::ForOp>(pop)) {
                pop.walk([&rewriter](memref::CopyOp copyOp) {
                  if (copyOp->hasAttr("lagrad_cache")) {
                    rewriter.eraseOp(copyOp);
                  }
                });
              }
              cloned->setAttr("cloned " + ctx.debug_names[pop.getResult(0)],
                              UnitAttr::get(pop.getContext()));
              for (auto tup :
                   llvm::zip(pop.getResults(), cloned->getResults())) {
                oldToCloned[std::get<0>(tup)] = std::get<1>(tup);
                ctx.debug_names[std::get<1>(tup)] =
                    "<cloned " + ctx.debug_names[std::get<0>(tup)] + ">";
              }
            }
          }
        }

        DenseMap<Value, Value> env;
        for (size_t i = 0; i < inputOperands.size(); i++) {
          env[inputOperands[i]] =
              iterArgs[iterArgs.size() - inputOperands.size() + i];
        }

        // This corresponds to all primal region iter args that are float/float
        // tensors
        SmallVector<Value> inputRegionArgs;
        for (auto iterArg : forOp.getRegionIterArgs()) {
          // TODO: stop treating the result as a special case
          if (iterArg ==
              forOp.getRegionIterArgForOpOperand(
                  forOp.getOpOperandForResult(forOp.getResults()[result_idx]))) {
            vjp_op = iterArg;
          } else if (isFloatOrFloatTensor(iterArg.getType())) {
            // Need to map again from gradient spaces from op operands to iter
            // args. This definitely feels brittle and should be cleaned up.
            env[iterArg] =
                env[forOp.getOpOperandForRegionIterArg(iterArg).get()];
            inputRegionArgs.push_back(iterArg);
          }
        }

        SmallVector<Operation *> reversedPrimalOps;
        for (auto &pop : primalOps) {
          reversedPrimalOps.push_back(&pop);
        }

        for (auto it = reversedPrimalOps.rbegin();
             it != reversedPrimalOps.rend(); it++) {
          auto &op = **it;
          if (auto yieldOp = dyn_cast_or_null<scf::YieldOp>(&op)) {
            for (size_t i = 0; i < yieldOp.getNumOperands(); i++) {
              Value operand = yieldOp.getOperand(i);
              if (i == result_idx) {
                env[operand] = iterArgs[0];
              } else if (isFloatOrFloatTensor(operand.getType())) {
                env[operand] =
                    iterArgs[iterArgs.size() - op.getNumOperands() + i];
                assert(operand.getType() == env[operand].getType() &&
                       "iter arg for grad space had unexpected type");
              }
            }
            rewriter.setInsertionPoint(builder.getBlock(),
                                       builder.getInsertionPoint());
          } else {
            populateVJP(&op, ctx, env, rewriter);
          }
        }

        SmallVector<Value> adjointResults;
        if (env[forOp.getResult(result_idx)]) {
          adjointResults.push_back(env[forOp.getResult(result_idx)]);
        } else if (vjp_op && env[vjp_op]) {
          adjointResults.push_back(env[vjp_op]);
        } else {
          // The primal result was unused in the primal loop body.
          adjointResults.push_back(iterArgs[0]);
        }

        for (size_t i = 0; i < free_operands.size(); i++) {
          auto free_operand = free_operands[i];
          if (!env[free_operand]) {
            // Assume free_operand is not active.
            env[free_operand] =
                getZero(free_operand.getLoc(), free_operand, rewriter);
          }
          adjointResults.push_back(env[free_operand]);
        }
        for (size_t i = 0; i < inputRegionArgs.size(); i++) {
          auto inputArg = inputRegionArgs[i];
          if (!env[inputArg]) {
            env[inputArg] =
                iterArgs[iterArgs.size() - inputOperands.size() + i];
          }
          adjointResults.push_back(env[inputArg]);
        }
        rewriter.create<scf::YieldOp>(loc, adjointResults);
      });
  // Replace ops referring to the old arguments to the new operands
  adjointFor.walk([&](Operation *op) {
    for (size_t i = 0; i < op->getNumOperands(); i++)
      if (oldToCloned[op->getOperand(i)])
        op->setOperand(i, oldToCloned[op->getOperand(i)]);
  });
  adjointFor->setAttr("adjoint of " + ctx.debug_names[forOp.getResult(0)],
                      UnitAttr::get(forOp.getContext()));

  // The output argument is a special case here. The gradient of the primal
  // result should always be the first adjoint result by construction.
  // TODO: Change this
  outer_env[forOp.getIterOperands()[result_idx]] = adjointFor.getResult(0);
  for (auto result_pair :
       llvm::zip(inputOperands, adjointFor.getResults().drop_front(1))) {
    auto free_operand = std::get<0>(result_pair);
    auto result_vjp = std::get<1>(result_pair);
    // If the free operand already has a space in the gradient, the for op
    // will add to that space.
    outer_env[free_operand] = result_vjp;
  }
}

void reverseForOpV1(scf::ForOp forOp, LAGradContext &ctx,
                    ValueRange free_operands, Value vjp_value,
                    size_t result_idx, DenseMap<Value, Value> &outer_env,
                    ConversionPatternRewriter &rewriter) {
  forOp.emitWarning() << "Using old style scf.for differentiation";
  PatternRewriter::InsertionGuard insertionGuard(rewriter);
  // Record the ops to clone before augmenting the primal with the caches.
  auto primalOps = forOp.getLoopBody().getOps();
  SmallVector<std::pair<Value, Value>> iterArgsToCached;
  DEPRECATEDpopulatePrimalCache(forOp, rewriter, iterArgsToCached);
  SmallVector<Value> operandsWithIV{
      forOp.getInductionVar(),
      // This is only valid under certain conditions, i.e. if the result was
      // only read once.
      forOp.getRegionIterArgs()[result_idx]};
  // TODO: This would be cleaner if the iter_args order was preserved, and free
  // operands were added after.
  // it goes [result value, ...free_operands, ...primal_iter_args]
  SmallVector<Value> iterArgsInit({vjp_value});
  // By construction, free operands come before iter arg grads, which is a
  // little awkward.
  SmallVector<Value> inputOperands{free_operands};
  inputOperands.reserve(free_operands.size() + forOp.getNumIterOperands());
  for (size_t i = 0; i < forOp.getNumIterOperands(); i++) {
    auto iterOperand = forOp.getIterOperands()[i];
    if (isFloatOrFloatTensor(iterOperand.getType()) && i != result_idx) {
      inputOperands.push_back(iterOperand);
    }
  }

  for (auto input_operand : inputOperands) {
    // Allocate spaces for the gradients of each input operand, if required.
    auto space = outer_env[input_operand]
                     ? outer_env[input_operand]
                     : getZero(input_operand.getLoc(), input_operand, rewriter,
                               /*init=*/true);
    iterArgsInit.push_back(space);
  }
  auto adjointFor = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      iterArgsInit,
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
        SmallVector<Value> regionArgs;
        Value idx = builder.create<arith::SubIOp>(loc, forOp.getUpperBound(), iv);
        Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
        idx = builder.create<arith::AddIOp>(loc, idx, forOp.getLowerBound());
        idx = builder.create<arith::SubIOp>(loc, idx, one);
        regionArgs.push_back(idx);
        regionArgs.push_back(forOp.getResult(result_idx));
        SmallVector<Value> replacedPrimalIterArgs;
        Value vjp_op = nullptr;
        for (auto cpair : iterArgsToCached) {
          operandsWithIV.push_back(cpair.first);
          Value loaded;
          if (auto tensorType =
                  cpair.first.getType().dyn_cast_or_null<RankedTensorType>()) {
            auto slice_layout =
                getRankReduceSubviewLayout(tensorType.getRank(), rewriter);
            auto resultType =
                MemRefType::get(tensorType.getShape(),
                                tensorType.getElementType(), slice_layout);
            auto intToAttr = [&](int64_t i) {
              return IntegerAttr::get(
                  IntegerType::get(builder.getContext(), 64), i);
            };
            SmallVector<Attribute> staticOffsets{
                IntegerAttr::get(IntegerType::get(builder.getContext(), 64),
                                 -9223372036854775808ULL)};
            SmallVector<Attribute> staticSizes{intToAttr(1)};
            SmallVector<Attribute> staticStrides{intToAttr(1)};
            for (int i = 0; i < tensorType.getRank(); i++) {
              staticOffsets.push_back(intToAttr(0));
              staticSizes.push_back(intToAttr(tensorType.getShape()[i]));
              staticStrides.push_back(intToAttr(1));
            }
            auto staticOffset =
                ArrayAttr::get(builder.getContext(), staticOffsets);
            auto staticSize = ArrayAttr::get(builder.getContext(), staticSizes);
            auto staticStride =
                ArrayAttr::get(builder.getContext(), staticStrides);

            auto view = builder.create<memref::SubViewOp>(
                cpair.second.getLoc(), resultType, cpair.second,
                /*dynamic shapes=*/ValueRange(idx), ValueRange(), ValueRange(),
                /*staticShapes=*/staticOffset, staticSize, staticStride);

            constexpr bool alloc_new = false;
            if (alloc_new) {
              auto dest = builder.create<memref::AllocOp>(
                  cpair.second.getLoc(),
                  MemRefType::get(tensorType.getShape(),
                                  tensorType.getElementType()));
              builder.create<memref::CopyOp>(cpair.second.getLoc(),
                                             view.getResult(), dest);

              loaded = builder.create<bufferization::ToTensorOp>(
                  cpair.second.getLoc(), dest.getResult());
            } else {
              // I don't know that this is always safe
              auto casted = builder.create<memref::CastOp>(
                  cpair.second.getLoc(), view.getResult(),
                  MemRefType::get(tensorType.getShape(),
                                  tensorType.getElementType()));
              loaded = builder.create<bufferization::ToTensorOp>(
                  cpair.second.getLoc(), casted.getResult());
            }
          } else {
            loaded = builder.create<memref::LoadOp>(cpair.second.getLoc(),
                                                    cpair.second, idx);
          }

          // This is to fix the case where the vjp value must be updated in the
          // body of the adjoint loop. TODO: This might not work with vectors
          if (cpair.first ==
              forOp.getRegionIterArgForOpOperand(
                  forOp.getOpOperandForResult(forOp.getResults()[result_idx]))) {
            vjp_op = loaded;
          } else if (isFloatOrFloatTensor(loaded.getType())) {
            replacedPrimalIterArgs.push_back(loaded);
          }
          regionArgs.push_back(loaded);
        }
        auto primalRegionOps = cloneBasicBlock(
            primalOps, builder, /*new=*/regionArgs, /*old=*/operandsWithIV,
            /*offsetInputs=*/false, &ctx);

        DenseMap<Value, Value> env;
        for (size_t i = 0; i < inputOperands.size(); i++) {
          env[inputOperands[i]] =
              iterArgs[iterArgs.size() - inputOperands.size() + i];
        }
        // This line is necessary because the copied ops in the primal (that we
        // iterate over in reverse) will have their operands replaced with
        // cached values, so we need this to make sure the gradient signal goes
        // to the right place.
        auto inputRegionArgs = ValueRange(replacedPrimalIterArgs);
        for (size_t i = 0; i < inputRegionArgs.size(); i++) {
          auto iterArg = iterArgs[iterArgs.size() - inputRegionArgs.size() + i];
          env[inputRegionArgs[i]] = iterArg;
          assert(env[inputRegionArgs[i]].getType() == iterArg.getType() &&
                 "reverseForOp: mismatched type when populating primal iter "
                 "arg gradient");
        }
        SmallVector<Value> adjointResults;
        for (auto it = primalRegionOps.rbegin(); it != primalRegionOps.rend();
             it++) {
          auto op = *it;
          auto opName = op->getName().getStringRef();
          if (opName == "scf.yield") {
            for (size_t i = 0; i < op->getNumOperands(); i++) {
              Value operand = op->getOperand(i);
              if (i == result_idx) {
                env[operand] = iterArgs[0];
              } else if (isFloatOrFloatTensor(operand.getType())) {
                env[operand] =
                    iterArgs[iterArgs.size() - op->getNumOperands() + i];
                assert(operand.getType() == env[operand].getType() &&
                       "iter arg for grad space had unexpected type");
              }
            }
            rewriter.setInsertionPointAfter(op);
            rewriter.eraseOp(op);
          } else {
            populateVJP(op, ctx, env, rewriter);
          }
        }

        if (env[forOp.getResult(result_idx)]) {
          adjointResults.push_back(env[forOp.getResult(result_idx)]);
        } else if (vjp_op && env[vjp_op]) {
          adjointResults.push_back(env[vjp_op]);
        } else {
          // The primal result was unused in the primal loop body.
          adjointResults.push_back(iterArgs[0]);
        }

        for (size_t i = 0; i < free_operands.size(); i++) {
          auto free_operand = free_operands[i];
          if (!env[free_operand]) {
            // Assume free_operand is not active.
            env[free_operand] =
                getZero(free_operand.getLoc(), free_operand, rewriter);
          }
          adjointResults.push_back(env[free_operand]);
        }
        for (size_t i = 0; i < inputRegionArgs.size(); i++) {
          auto inputArg = inputRegionArgs[i];
          if (!env[inputArg]) {
            env[inputArg] =
                iterArgs[iterArgs.size() - inputOperands.size() + i];
          }
          adjointResults.push_back(env[inputArg]);
        }
        rewriter.create<scf::YieldOp>(loc, adjointResults);
      });

  // The output argument is a special case here. The gradient of the primal
  // result should always be the first adjoint result by construction.
  outer_env[forOp.getIterOperands()[result_idx]] = adjointFor.getResult(0);
  for (auto result_pair :
       llvm::zip(inputOperands, adjointFor.getResults().drop_front(1))) {
    auto free_operand = std::get<0>(result_pair);
    auto result_vjp = std::get<1>(result_pair);
    // If the free operand already has a space in the gradient, the for op
    // will add to that space.
    outer_env[free_operand] = result_vjp;
  }
}

Value reverseIfOpV2(scf::IfOp ifOp, LAGradContext &ctx, Value freeOperand,
                    Value vjp_value, DenseMap<Value, Value> outer_env,
                    ConversionPatternRewriter &rewriter) {
  // Find parent iter args to determine which primal ops can be safely cloned
  // without dominance issues.
  ValueSet parentIterArgs;
  auto parent = ifOp->getParentOfType<scf::ForOp>();
  while (parent) {
    for (auto iterArg : parent.getRegionIterArgs()) {
      parentIterArgs.insert(iterArg);
    }
    parent = parent->getParentOfType<scf::ForOp>();
  }

  SmallVector<Value> caches;
  // Ignore any values that depend directly on the caches
  for (auto pair : ctx.tbrCachedVals) {
    caches.push_back(pair.second);
  }

  DenseMap<Value, Value> oldToCloned;
  auto reverseIfBlock = [&](Region &ifRegion) {
    return [&](OpBuilder &builder, Location loc) {
      PatternRewriter::InsertionGuard insertionGuard(rewriter);
      // Clone ops that don't depend on the region iter args
      for (auto &pop : ifRegion.getOps()) {
        if (pop.getNumResults() > 0) {
          SmallVector<Value> frontier{pop.getResults()};
          ValueSet deps;
          runBottomUpDFS(frontier, deps);
          auto inDependSet = [&](Value v) { return deps.contains(v); };
          if (!(llvm::any_of(caches, inDependSet) ||
                llvm::any_of(parentIterArgs, inDependSet))) {
            auto cloned = builder.clone(pop);
            for (auto tup : llvm::zip(pop.getResults(), cloned->getResults())) {
              oldToCloned[std::get<0>(tup)] = std::get<1>(tup);
              ctx.debug_names[std::get<1>(tup)] =
                  "<cloned " + ctx.debug_names[std::get<0>(tup)] + ">";
            }
          }
        }
      }

      DenseMap<Value, Value> env;
      SmallVector<Operation *> reversedPrimalOps;
      for (auto &pop : ifRegion.getOps()) {
        reversedPrimalOps.push_back(&pop);
      }

      for (auto it = reversedPrimalOps.rbegin(); it != reversedPrimalOps.rend();
           it++) {
        auto &op = **it;
        if (auto yieldOp = dyn_cast_or_null<scf::YieldOp>(&op)) {
          // assert(yieldOp.getNumOperands() == 1 &&
          //        "expected scf.yield in scf.if to have one operand");
          Value operand = yieldOp.getOperand(0);
          env[operand] = vjp_value;
          rewriter.setInsertionPoint(builder.getBlock(),
                                     builder.getInsertionPoint());
        } else {
          populateVJP(&op, ctx, env, rewriter);
        }
      }
      // The free operand might only appear in one block but not the other.
      if (!env[freeOperand]) {
        rewriter.create<scf::YieldOp>(loc, getZero(loc, freeOperand, rewriter));
      } else {
        rewriter.create<scf::YieldOp>(loc, env[freeOperand]);
      }
    };
  };
  auto adjointIf = rewriter.create<scf::IfOp>(
      ifOp->getLoc(), /*resultTypes=*/freeOperand.getType(),
      /*cond=*/ifOp.getCondition(),
      /*thenBuilder=*/reverseIfBlock(ifOp.getThenRegion()),
      /*elseBuilder=*/reverseIfBlock(ifOp.getElseRegion()));
  // Replace ops referring to the old arguments to the new operands
  adjointIf.walk([&](Operation *op) {
    for (size_t i = 0; i < op->getNumOperands(); i++)
      if (oldToCloned[op->getOperand(i)])
        op->setOperand(i, oldToCloned[op->getOperand(i)]);
  });
  return adjointIf.getResult(0);
}

Value reverseIfOpV1(scf::IfOp ifOp, LAGradContext &ctx, Value freeOperand,
                    Value vjp_value, DenseMap<Value, Value> outer_env,
                    ConversionPatternRewriter &rewriter) {
  auto reverseIfBlock = [&](Region &ifRegion) {
    return [&](OpBuilder &builder, Location loc) {
      PatternRewriter::InsertionGuard insertionGuard(rewriter);
      auto primalRegionOps = cloneBasicBlock(ifRegion.getOps(), builder, {}, {},
                                             /*offsetInputs=*/false, &ctx);
      DenseMap<Value, Value> env;
      for (auto it = primalRegionOps.rbegin(); it != primalRegionOps.rend();
           it++) {
        auto op = *it;
        auto opName = op->getName().getStringRef();
        if (opName == "scf.yield") {
          Value operand = op->getOperand(0);
          env[operand] = vjp_value;
          rewriter.setInsertionPointAfter(op);
          rewriter.eraseOp(op);
        } else {
          populateVJP(op, ctx, env, rewriter);
        }
      }
      // The free operand might only appear in one block but not the other.
      if (!env[freeOperand]) {
        rewriter.create<scf::YieldOp>(loc, getZero(loc, freeOperand, rewriter));
      } else {
        rewriter.create<scf::YieldOp>(loc, env[freeOperand]);
      }
    };
  };

  auto adjointIf = rewriter.create<scf::IfOp>(
      ifOp->getLoc(), /*resultTypes=*/freeOperand.getType(),
      /*cond=*/ifOp.getCondition(),
      /*thenBuilder=*/reverseIfBlock(ifOp.getThenRegion()),
      /*elseBuilder=*/reverseIfBlock(ifOp.getElseRegion()));
  return adjointIf.getResult(0);
}

Value reverseIfOp(scf::IfOp ifOp, LAGradContext &ctx, Value freeOperand,
                  Value vjp_value, DenseMap<Value, Value> outer_env,
                  ConversionPatternRewriter &rewriter) {
  return reverseIfOpV2(ifOp, ctx, freeOperand, vjp_value, outer_env, rewriter);
}

void reverseForOp(scf::ForOp forOp, LAGradContext &ctx,
                  ValueRange free_operands, Value vjp_value, size_t result_idx,
                  DenseMap<Value, Value> &outer_env,
                  ConversionPatternRewriter &rewriter) {
  reverseForOpV2(forOp, ctx, free_operands, vjp_value, result_idx, outer_env,
                 rewriter);
}

} // namespace mlir
