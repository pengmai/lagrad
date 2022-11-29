#include "LAGrad/LAGradDialect.h"
#include "LAGrad/LAGradOps.h"
#include "LAGrad/Transforms.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace lagrad;

using llvm::errs;
namespace {

/// Forward mode results in creating ops in the same region. We need to save
/// them to avoid trying to differentiate generated adjoint ops.
SmallVector<Operation *> savePrimalOps(Region *region) {
  size_t numOps =
      std::distance(region->getOps().begin(), region->getOps().end());
  SmallVector<Operation *> primalOps{numOps - 1};
  llvm::transform(region->front().without_terminator(), primalOps.begin(),
                  [](Operation &op) { return &op; });
  return primalOps;
}

// TODO: Organize forward mode functions better.
LogicalResult populateJVP(Operation *op, LAGradContext &ctx,
                          BlockAndValueMapping &env,
                          ConversionPatternRewriter &rewriter);

LogicalResult linalgJVP(linalg::LinalgOp op, LAGradContext &ctx,
                        BlockAndValueMapping &env,
                        ConversionPatternRewriter &rewriter) {
  SmallVector<Value> inputs, outputs;
  SmallVector<Type> outputTypes;
  SmallVector<AffineMap> indexingMaps;
  DenseMap<unsigned, unsigned> dualMapping;
  SmallVector<std::pair<unsigned, unsigned>> replacedPrimalMapping;

  SmallVector<StringRef> iteratorTypes{op.iterator_types().size()};
  llvm::transform(
      op.iterator_types(), iteratorTypes.begin(),
      [](Attribute attr) { return attr.cast<StringAttr>().getValue(); });

  unsigned idx = 0;
  unsigned origIdx = 0;
  for (OpOperand *input : op.getInputOperands()) {
    inputs.push_back(input->get());
    indexingMaps.push_back(op.getTiedIndexingMap(input));
    replacedPrimalMapping.push_back(std::make_pair(origIdx, idx));
    if (env.contains(input->get())) {
      inputs.push_back(env.lookup(input->get()));
      indexingMaps.push_back(op.getTiedIndexingMap(input));

      dualMapping[idx] = idx + 1;
      ++idx;
    }
    ++idx;
    ++origIdx;
  }

  for (OpOperand *output : op.getOutputOperands()) {
    outputs.push_back(output->get());
    outputTypes.push_back(output->get().getType());
    indexingMaps.push_back(op.getTiedIndexingMap(output));
    replacedPrimalMapping.push_back(std::make_pair(origIdx, idx));
    if (env.contains(output->get())) {
      Value dual = env.lookup(output->get());
      outputs.push_back(dual);
      outputTypes.push_back(dual.getType());
      indexingMaps.push_back(op.getTiedIndexingMap(output));
      dualMapping[idx] = idx + 1;
      ++idx;
    }
    ++idx;
    ++origIdx;
  }

  auto terminator = cast<linalg::YieldOp>(op.getBlock()->getTerminator());
  BlockAndValueMapping map;
  bool augmentFailed = false;
  auto newOp = rewriter.create<linalg::GenericOp>(
      op.getLoc(), outputTypes, inputs, outputs, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange regionArgs) {
        PatternRewriter::InsertionGuard guard(rewriter);
        auto startOfBlock = b.saveInsertionPoint();
        rewriter.restoreInsertionPoint(startOfBlock);
        for (auto mapping : dualMapping) {
          env.map(regionArgs[mapping.first], regionArgs[mapping.second]);
        }

        for (auto pair : replacedPrimalMapping) {
          map.map(op.getBlock()->getArgument(pair.first),
                  regionArgs[pair.second]);
        }

        // Clone over original ops.
        SmallVector<Operation *> bodyOps;
        for (Operation &origOp : op.getBlock()->without_terminator()) {
          Operation *clonedOp = b.clone(origOp, map);
          bodyOps.push_back(clonedOp);
          map.map(origOp.getResults(), clonedOp->getResults());
        }

        // Generate JVPs.
        for (auto *bodyOp : bodyOps) {
          if (failed(populateJVP(bodyOp, ctx, env, rewriter))) {
            augmentFailed = true;
            return;
          }
        }

        // Find yield operands.
        SmallVector<Value> results;
        results.reserve(outputs.size());
        for (auto mapping : replacedPrimalMapping) {
          if (mapping.first >= op.getNumInputs()) {
            Value newYieldOperand = map.lookup(
                terminator.getOperand(mapping.first - op.getNumInputs()));
            results.push_back(newYieldOperand);
            if (dualMapping.count(mapping.second))
              results.push_back(env.lookup(newYieldOperand));
          }
        }
        b.create<linalg::YieldOp>(loc, results);
      });
  if (augmentFailed) {
    return failure();
  }

  for (auto mapping : replacedPrimalMapping) {
    if (mapping.first >= op.getNumInputs()) {
      op->getResult(mapping.first - op.getNumInputs())
          .replaceAllUsesWith(
              newOp.getResult(mapping.second - newOp.getNumInputs()));
    }
  }
  for (auto mapping : dualMapping) {
    env.map(newOp.getResult(mapping.first - newOp.getNumInputs()),
            newOp.getResult(mapping.second - newOp.getNumInputs()));
  }
  rewriter.eraseOp(op);
  return success();
}

LogicalResult forLoopJVP(scf::ForOp forOp, LAGradContext &ctx,
                         BlockAndValueMapping &env,
                         ConversionPatternRewriter &rewriter) {
  SmallVector<Value> iterArgInits;
  DenseMap<unsigned, unsigned> dualMapping;
  SmallVector<std::pair<unsigned, unsigned>> replacedPrimalMapping;
  unsigned idx = 0;
  unsigned originalIdx = 0;
  for (Value iterOperand : forOp.getIterOperands()) {
    iterArgInits.push_back(iterOperand);
    replacedPrimalMapping.push_back(std::make_pair(originalIdx, idx));
    if (env.contains(iterOperand)) {
      iterArgInits.push_back(env.lookup(iterOperand));
      dualMapping[idx] = idx + 1;
      ++idx;
    }
    ++idx;
    ++originalIdx;
  }

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  // Create a new for op because we need to pass additional iteration args
  // with the dual numbers.
  BlockAndValueMapping map;
  bool augmentFailed = false;
  auto augmentedFor = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.lowerBound(), forOp.upperBound(), forOp.step(),
      iterArgInits,
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange regionArgs) {
        PatternRewriter::InsertionGuard guard(rewriter);
        auto startOfBlock = builder.saveInsertionPoint();
        rewriter.restoreInsertionPoint(startOfBlock);

        // The region arg mapping must match the iter operand mapping.
        for (auto mapping : dualMapping) {
          env.map(regionArgs[mapping.first], regionArgs[mapping.second]);
        }
        for (auto mapping : replacedPrimalMapping) {
          map.map(forOp.getRegionIterArgs()[mapping.first],
                  regionArgs[mapping.second]);
        }
        map.map(forOp.getInductionVar(), iv);

        // Clone over the original ops.
        SmallVector<Operation *> bodyOps;
        for (Operation &origOp : forOp.getBody()->without_terminator()) {
          Operation *clonedOp = builder.clone(origOp, map);
          bodyOps.push_back(clonedOp);
          map.map(origOp.getResults(), clonedOp->getResults());
        }

        // Generate JVPs.
        for (auto *bodyOp : bodyOps) {
          if (failed(populateJVP(bodyOp, ctx, env, rewriter))) {
            augmentFailed = true;
            return;
          }
        }

        SmallVector<Value> results;
        results.reserve(regionArgs.size());
        for (auto mapping : replacedPrimalMapping) {
          Value newYieldOperand = map.lookup(yieldOp.getOperand(mapping.first));
          results.push_back(newYieldOperand);
          if (dualMapping.count(mapping.second))
            results.push_back(env.lookup(newYieldOperand));
        }
        rewriter.create<scf::YieldOp>(loc, results);
      });
  if (augmentFailed) {
    return failure();
  }

  for (auto mapping : replacedPrimalMapping) {
    forOp.getResult(mapping.first)
        .replaceAllUsesWith(augmentedFor.getResult(mapping.second));
  }
  for (auto mapping : dualMapping) {
    env.map(augmentedFor.getResult(mapping.first),
            augmentedFor.getResult(mapping.second));
  }
  rewriter.eraseOp(forOp);
  return success();
}

LogicalResult populateJVP(Operation *op, LAGradContext &ctx,
                          BlockAndValueMapping &env,
                          ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(op);
  Location loc = op->getLoc();
  Value jvp;
  if (op->getNumResults() == 0) {
    return success();
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    return forLoopJVP(forOp, ctx, env, rewriter);
  } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    return linalgJVP(op, ctx, env, rewriter);
  }

  assert(op->getNumResults() == 1);
  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    if (isFloatOrFloatTensor(constOp.getType())) {
      jvp = getZero(constOp.getLoc(), constOp.getResult(), rewriter);
    } else if (auto attr = constOp.value().dyn_cast<FloatAttr>()) {
      // constOp.getType().getIntOrFloatBitWidth()
      jvp = rewriter.create<arith::ConstantOp>(loc,
                                               rewriter.getF64FloatAttr(0.0));
    } else {
      return success();
    }
  } else if (auto addfOp = dyn_cast<arith::AddFOp>(op)) {
    jvp = rewriter.create<arith::AddFOp>(loc, env.lookup(addfOp.lhs()),
                                         env.lookup(addfOp.rhs()));
  } else if (auto mulfOp = dyn_cast<arith::MulFOp>(op)) {
    if (!env.contains(mulfOp.lhs())) {
      jvp = rewriter.create<arith::MulFOp>(loc, mulfOp.lhs(),
                                           env.lookup(mulfOp.rhs()));
    } else if (!env.contains(mulfOp.rhs())) {
      jvp = rewriter.create<arith::MulFOp>(loc, mulfOp.rhs(),
                                           env.lookup(mulfOp.lhs()));
    } else {
      jvp = rewriter.create<arith::AddFOp>(
          loc,
          rewriter.create<arith::MulFOp>(loc, mulfOp.rhs(),
                                         env.lookup(mulfOp.lhs())),
          rewriter.create<arith::MulFOp>(loc, mulfOp.lhs(),
                                         env.lookup(mulfOp.rhs())));
    }
  } else if (auto sinOp = dyn_cast<math::SinOp>(op)) {
    jvp = rewriter.create<arith::MulFOp>(
        loc, env.lookup(sinOp.getOperand()),
        rewriter.create<math::CosOp>(loc, sinOp.getOperand()));
  } else {
    op->emitOpError() << "unhandled op";
    return failure();
  }
  env.map(op->getResult(0), jvp);
  return success();
}

static LogicalResult generateTangent(FuncOp tangentFunc, LAGradContext &ctx,
                                     ConversionPatternRewriter &rewriter) {
  PatternRewriter::InsertionGuard guard(rewriter);
  Region *region = tangentFunc.getCallableRegion();
  if (!region) {
    tangentFunc.emitOpError("Cannot perform AD on op without body");
    return failure();
  }
  assert(region->hasOneBlock());

  // env maps primal values to their dual values.
  BlockAndValueMapping env;
  size_t idx = 0;
  for (BlockArgument arg : tangentFunc.getArguments()) {
    // good ol' bandaid for activity analysis.
    if (isFloatOrFloatTensor(arg.getType())) {
      ++idx;
      tangentFunc.insertArgument(idx, arg.getType(), {});
      env.map(arg, tangentFunc.getArgument(idx));
    }
    ++idx;
  }

  SmallVector<Operation *> primalOps = savePrimalOps(region);
  for (Operation *op : primalOps) {
    if (failed(populateJVP(op, ctx, env, rewriter))) {
      return failure();
    }
  }

  Operation *terminator = region->front().getTerminator();
  SmallVector<Value> results{terminator->getNumOperands()};
  llvm::transform(terminator->getOperands(), results.begin(),
                  [&env](Value operand) { return env.lookup(operand); });
  // tangentFunc.getType()
  rewriter.create<ReturnOp>(terminator->getLoc(), results);
  rewriter.eraseOp(terminator);
  return success();
}

class ForwardModeAD : public OpConversionPattern<TangentOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TangentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tangentFunc = generateTangentFunc(op, rewriter);
    if (!tangentFunc)
      return failure();

    rewriter.replaceOpWithNewOp<CallOp>(op, tangentFunc, op.getOperands());
    return success();
  }

private:
  static FuncOp generateTangentFunc(TangentOp op,
                                    ConversionPatternRewriter &rewriter) {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto originalFuncOp = moduleOp.lookupSymbol<FuncOp>(op.FAttr());
    std::string tangentFuncName =
        ("__tangent_" + originalFuncOp.getName()).str();
    if (auto existingFunc = moduleOp.lookupSymbol<FuncOp>(tangentFuncName))
      return existingFunc;

    FuncOp tangentFunc =
        copyFunctionDeclaration(originalFuncOp, tangentFuncName, rewriter);
    LAGradContext lagradctx{moduleOp};
    DEBUGpopulateFunc(lagradctx.debug_names, tangentFunc);
    if (failed(generateTangent(tangentFunc, lagradctx, rewriter))) {
      return nullptr;
    }
    return tangentFunc;
  }
};
} // namespace

void mlir::lagrad::populateLAGradTransforms(OwningRewritePatternList &patterns,
                                            MLIRContext *ctx) {
  patterns.add<ForwardModeAD>(ctx);
}
