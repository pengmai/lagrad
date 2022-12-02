#include "LAGrad/LAGradDialect.h"
#include "LAGrad/LAGradOps.h"
#include "LAGrad/Transforms.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace lagrad;
using utils::IteratorType;

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

void updateActiveValues(LAGradContext &ctx, ValueRange from, ValueRange to) {
  for (auto pair : llvm::zip(from, to)) {
    if (ctx.activeValues.contains(std::get<0>(pair))) {
      ctx.activeValues.insert(std::get<1>(pair));
    }
  }
}

LogicalResult linalgJVP(linalg::LinalgOp op, LAGradContext &ctx,
                        BlockAndValueMapping &env,
                        ConversionPatternRewriter &rewriter) {
  SmallVector<Value> inputs, outputs;
  SmallVector<Type> outputTypes;
  SmallVector<AffineMap> indexingMaps;
  DenseMap<unsigned, unsigned> dualMapping;
  SmallVector<std::pair<unsigned, unsigned>> replacedPrimalMapping;
  SmallVector<IteratorType> iteratorTypes{op.getIteratorTypesArray()};

  unsigned idx = 0;
  unsigned origIdx = 0;
  for (OpOperand *input : op.getDpsInputOperands()) {
    inputs.push_back(input->get());
    indexingMaps.push_back(op.getMatchingIndexingMap(input));
    replacedPrimalMapping.push_back(std::make_pair(origIdx, idx));
    if (env.contains(input->get())) {
      inputs.push_back(env.lookup(input->get()));
      indexingMaps.push_back(op.getMatchingIndexingMap(input));

      dualMapping[idx] = idx + 1;
      ++idx;
    }
    ++idx;
    ++origIdx;
  }

  for (OpOperand *output : op.getDpsInitOperands()) {
    outputs.push_back(output->get());
    outputTypes.push_back(output->get().getType());
    indexingMaps.push_back(op.getMatchingIndexingMap(output));
    replacedPrimalMapping.push_back(std::make_pair(origIdx, idx));
    if (env.contains(output->get())) {
      Value dual = env.lookup(output->get());
      outputs.push_back(dual);
      outputTypes.push_back(dual.getType());
      indexingMaps.push_back(op.getMatchingIndexingMap(output));
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
          if (mapping.first >= op.getNumDpsInputs()) {
            Value newYieldOperand = map.lookup(
                terminator.getOperand(mapping.first - op.getNumDpsInputs()));
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

  SmallVector<Value> replacedResults{static_cast<size_t>(op.getNumDpsInits())};
  for (auto mapping : replacedPrimalMapping) {
    if (mapping.first >= op.getNumDpsInputs()) {
      replacedResults[mapping.first - op.getNumDpsInputs()] =
          newOp.getResult(mapping.second - newOp.getNumDpsInputs());
    }
  }
  rewriter.replaceOp(op, replacedResults);

  for (auto mapping : dualMapping) {
    if (mapping.first >= newOp.getNumDpsInputs()) {
      env.map(newOp.getResult(mapping.first - newOp.getNumDpsInputs()),
              newOp.getResult(mapping.second - newOp.getNumDpsInputs()));
    }
  }
  return success();
}

LogicalResult ifJVP(scf::IfOp ifOp, LAGradContext &ctx,
                    BlockAndValueMapping &env,
                    ConversionPatternRewriter &rewriter) {
  Location loc = ifOp.getLoc();
  SmallVector<Type> resultTypes;
  SmallVector<std::pair<unsigned, unsigned>> replacedPrimalMapping;
  DenseMap<unsigned, unsigned> dualMapping;
  unsigned idx = 0;
  unsigned origIdx = 0;
  for (OpResult result : ifOp.getResults()) {
    resultTypes.push_back(result.getType());
    replacedPrimalMapping.push_back(std::make_pair(origIdx, idx));
    // FIXME: need proper activity analysis
    if (isFloatOrFloatTensor(result.getType())) {
      // if (ctx.activeValues.contains(result)) {
      resultTypes.push_back(result.getType());
      dualMapping[idx] = idx + 1;
      idx++;
    }
    idx++;
    origIdx++;
  }

  auto builderFunc = [&](Block &block) {
    return [&](OpBuilder &builder, Location loc) {
      PatternRewriter::InsertionGuard guard(rewriter);

      auto startOfBlock = builder.saveInsertionPoint();
      rewriter.restoreInsertionPoint(startOfBlock);
      BlockAndValueMapping map;

      // Clone over the original ops.
      SmallVector<Operation *> bodyOps;
      for (Operation &origOp : block.without_terminator()) {
        Operation *clonedOp = builder.clone(origOp, map);
        bodyOps.push_back(clonedOp);
        map.map(origOp.getResults(), clonedOp->getResults());
        for (OpResult r : origOp.getResults()) {
          if (env.contains(r)) {
            env.map(clonedOp->getResult(r.getResultNumber()), env.lookup(r));
          }
        }
      }

      // Generate JVPs.
      for (auto *bodyOp : bodyOps) {
        if (failed(populateJVP(bodyOp, ctx, env, rewriter))) {
          llvm_unreachable("populate jvp failed inside if op\n");
          return;
        }
      }
      SmallVector<Value> results;
      auto yieldOp = cast<scf::YieldOp>(block.getTerminator());

      for (auto mapping : replacedPrimalMapping) {
        Value newYieldOperand = rewriter.getRemappedValue(
            map.lookupOrDefault(yieldOp.getOperand(mapping.first)));
        results.push_back(newYieldOperand);
        if (dualMapping.count(mapping.second)) {
          results.push_back(env.lookupOrDefault(newYieldOperand));
        }
      }
      rewriter.create<scf::YieldOp>(loc, results);
    };
  };

  auto augmentedIf = rewriter.create<scf::IfOp>(
      loc, resultTypes, ifOp.getCondition(), builderFunc(*ifOp.thenBlock()),
      builderFunc(*ifOp.elseBlock()));
  SmallVector<Value> replacedResults{replacedPrimalMapping.size()};
  for (auto mapping : replacedPrimalMapping) {
    replacedResults[mapping.first] = augmentedIf.getResult(mapping.second);
    ifOp.getResult(mapping.first)
        .replaceAllUsesWith(augmentedIf.getResult(mapping.second));
  }
  rewriter.replaceOp(ifOp, replacedResults);
  for (auto mapping : dualMapping) {
    env.map(augmentedIf.getResult(mapping.first),
            augmentedIf.getResult(mapping.second));
  }
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
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), iterArgInits,
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
          updateActiveValues(ctx, origOp.getResults(), clonedOp->getResults());
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
          Value newYieldOperand = rewriter.getRemappedValue(
              map.lookup(yieldOp.getOperand(mapping.first)));
          results.push_back(newYieldOperand);
          if (dualMapping.count(mapping.second))
            results.push_back(env.lookup(newYieldOperand));
        }
        rewriter.create<scf::YieldOp>(loc, results);
      });
  if (augmentFailed) {
    return failure();
  }

  SmallVector<Value> replacedResults{replacedPrimalMapping.size()};
  for (auto mapping : replacedPrimalMapping) {
    replacedResults[mapping.first] = augmentedFor.getResult(mapping.second);
    forOp.getResult(mapping.first)
        .replaceAllUsesWith(augmentedFor.getResult(mapping.second));
  }
  rewriter.replaceOp(forOp, replacedResults);
  for (auto mapping : dualMapping) {
    env.map(augmentedFor.getResult(mapping.first),
            augmentedFor.getResult(mapping.second));
  }
  // rewriter.eraseOp(forOp);
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
  // Need to hit these before the linalg op interface.
  if (auto tensorEmptyOp = dyn_cast<tensor::EmptyOp>(op)) {
    jvp = rewriter.create<tensor::EmptyOp>(
        loc, tensorEmptyOp.getMixedSizes(),
        tensorEmptyOp.getType().getElementType());
    env.map(tensorEmptyOp.getResult(), jvp);
    return success();
  } else if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
    auto dualFillOp =
        rewriter.create<linalg::FillOp>(loc, fillOp.value(), fillOp.output());
    env.map(fillOp.getResults(), dualFillOp.getResults());
    return success();
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    return ifJVP(ifOp, ctx, env, rewriter);
  } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    return forLoopJVP(forOp, ctx, env, rewriter);
  } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    return linalgJVP(op, ctx, env, rewriter);
  }

  assert(op->getNumResults() == 1);
  if (op->getResult(0).getType().isIntOrIndex()) {
    return success();
  }

  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    if (isFloatOrFloatTensor(constOp.getType())) {
      jvp = getZero(constOp.getLoc(), constOp.getResult(), rewriter);
    } else if (auto attr = constOp.getValue().dyn_cast<FloatAttr>()) {
      // constOp.getType().getIntOrFloatBitWidth()
      jvp = rewriter.create<arith::ConstantOp>(loc,
                                               rewriter.getF64FloatAttr(0.0));
    } else {
      return success();
    }
  } else if (auto addfOp = dyn_cast<arith::AddFOp>(op)) {
    if (!env.contains(addfOp.getLhs())) {
      jvp = env.lookup(addfOp.getRhs());
    } else if (!env.contains(addfOp.getRhs())) {
      jvp = env.lookup(addfOp.getLhs());
    } else {
      jvp = rewriter.create<arith::AddFOp>(loc, env.lookup(addfOp.getLhs()),
                                           env.lookup(addfOp.getRhs()));
    }
  } else if (auto subfOp = dyn_cast<arith::SubFOp>(op)) {
    jvp = rewriter.create<arith::SubFOp>(loc, env.lookup(subfOp.getLhs()),
                                         env.lookup(subfOp.getRhs()));
  } else if (auto mulfOp = dyn_cast<arith::MulFOp>(op)) {
    if (!env.contains(mulfOp.getLhs())) {
      jvp = rewriter.create<arith::MulFOp>(loc, mulfOp.getLhs(),
                                           env.lookup(mulfOp.getRhs()));
    } else if (!env.contains(mulfOp.getRhs())) {
      jvp = rewriter.create<arith::MulFOp>(loc, mulfOp.getRhs(),
                                           env.lookup(mulfOp.getLhs()));
    } else {
      jvp = rewriter.create<arith::AddFOp>(
          loc,
          rewriter.create<arith::MulFOp>(loc, mulfOp.getRhs(),
                                         env.lookup(mulfOp.getLhs())),
          rewriter.create<arith::MulFOp>(loc, mulfOp.getLhs(),
                                         env.lookup(mulfOp.getRhs())));
    }
  } else if (auto negfOp = dyn_cast<arith::NegFOp>(op)) {
    jvp = rewriter.create<arith::NegFOp>(loc, env.lookup(negfOp.getOperand()));
  } else if (auto divfOp = dyn_cast<arith::DivFOp>(op)) {
    Value lhsDual = rewriter.create<arith::DivFOp>(
        loc, env.lookup(divfOp.getLhs()), divfOp.getRhs());

    // RHS
    jvp = rewriter.create<arith::MulFOp>(
        op->getLoc(), env.lookup(divfOp.getRhs()), divfOp.getLhs());
    jvp = rewriter.create<arith::NegFOp>(op->getLoc(), jvp);
    Value denom = rewriter.create<arith::MulFOp>(op->getLoc(), divfOp.getRhs(),
                                                 divfOp.getRhs());
    Value rhsDual = rewriter.create<arith::DivFOp>(op->getLoc(), jvp, denom);

    jvp = rewriter.create<arith::AddFOp>(loc, lhsDual, rhsDual);
  } else if (auto sqrtOp = dyn_cast<math::SqrtOp>(op)) {
    auto half = constLike(loc, sqrtOp.getOperand(), 0.5, rewriter);
    jvp = rewriter.create<arith::DivFOp>(
        loc,
        rewriter.create<arith::MulFOp>(loc, env.lookup(sqrtOp.getOperand()),
                                       half),
        sqrtOp.getResult());
  } else if (auto sinOp = dyn_cast<math::SinOp>(op)) {
    jvp = rewriter.create<arith::MulFOp>(
        loc, env.lookup(sinOp.getOperand()),
        rewriter.create<math::CosOp>(loc, sinOp.getOperand()));
  } else if (auto cosOp = dyn_cast<math::CosOp>(op)) {
    jvp = rewriter.create<arith::MulFOp>(
        loc, env.lookup(cosOp.getOperand()),
        rewriter.create<arith::NegFOp>(
            loc, rewriter.create<math::SinOp>(loc, cosOp.getOperand())));
  } else if (auto insertOp = dyn_cast<tensor::InsertOp>(op)) {
    jvp = rewriter.create<tensor::InsertOp>(
        loc, env.lookup(insertOp.getScalar()), env.lookup(insertOp.getDest()),
        insertOp.getIndices());
  } else if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
    jvp = rewriter.create<tensor::ExtractOp>(
        loc, env.lookup(rewriter.getRemappedValue(extractOp.getTensor())),
        extractOp.getIndices());
  } else if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
    // Need to refactor the lookups
    // into a getDual
    jvp = rewriter.create<tensor::InsertSliceOp>(
        loc, env.lookup(rewriter.getRemappedValue(insertSliceOp.getSource())),
        env.lookup(insertSliceOp.getDest()), insertSliceOp.getMixedOffsets(),
        insertSliceOp.getMixedSizes(), insertSliceOp.getMixedStrides());
  } else if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    jvp = rewriter.create<tensor::ExtractSliceOp>(
        loc, extractSliceOp.getType(),
        env.lookup(rewriter.getRemappedValue(extractSliceOp.getSource())),
        extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
        extractSliceOp.getMixedStrides());
  } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
    // for (auto operand : callOp.getOperands()) {
    //   errs() << "call op operand: " << operand << "\n";
    //   errs() << "call op operand is active: "
    //          << ctx.activeValues.contains(operand) << "\n";
    // }
    errs() << "call op jvp in progress\n";
    return failure();
  } else {
    op->emitOpError() << "unhandled op";
    return failure();
  }
  env.map(op->getResult(0), jvp);
  return success();
}

static LogicalResult generateTangent(func::FuncOp tangentFunc,
                                     LAGradContext &ctx,
                                     ArrayAttr activeArgsAttr,
                                     ConversionPatternRewriter &rewriter) {
  PatternRewriter::InsertionGuard guard(rewriter);
  Region *region = tangentFunc.getCallableRegion();
  if (!region) {
    tangentFunc.emitOpError("Cannot perform AD on op without body");
    return failure();
  }
  assert(region->hasOneBlock());
  rewriter.setInsertionPointToStart(&region->front());

  // env maps primal values to their dual values.
  BlockAndValueMapping env;
  size_t idx = 0;
  DenseSet<size_t> activeArgs;
  if (!activeArgsAttr || activeArgsAttr.empty()) {
    activeArgs.insert(0);
  }
  for (APInt attr : activeArgsAttr.getAsValueRange<IntegerAttr>()) {
    activeArgs.insert(attr.getSExtValue());
  }
  SmallVector<BlockArgument> originalArgs{tangentFunc.getArguments().begin(),
                                          tangentFunc.getArguments().end()};
  for (BlockArgument arg : originalArgs) {
    if (activeArgs.contains(arg.getArgNumber())) {
      ++idx;
      tangentFunc.insertArgument(idx, arg.getType(), {}, tangentFunc.getLoc());
      env.map(arg, tangentFunc.getArgument(idx));
    } else if (isFloatOrFloatTensor(arg.getType())) {
      // TODO: modify populateJVP to not need these dummy zero values.
      env.map(arg, getZero(tangentFunc.getLoc(), arg, rewriter));
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
                  [&](Value operand) {
                    return env.lookup(rewriter.getRemappedValue(operand));
                  });
  // tangentFunc.getType()
  rewriter.create<func::ReturnOp>(terminator->getLoc(), results);
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

    rewriter.replaceOpWithNewOp<func::CallOp>(op, tangentFunc,
                                              op.getOperands());
    return success();
  }

private:
  static func::FuncOp generateTangentFunc(TangentOp op,
                                          ConversionPatternRewriter &rewriter) {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto originalFuncOp = moduleOp.lookupSymbol<func::FuncOp>(op.getFAttr());
    std::string tangentFuncName =
        ("__tangent_" + originalFuncOp.getName()).str();
    if (auto existingFunc =
            moduleOp.lookupSymbol<func::FuncOp>(tangentFuncName))
      return existingFunc;

    func::FuncOp tangentFunc =
        copyFunctionDeclaration(originalFuncOp, tangentFuncName, rewriter);
    auto tangentOf = op->getAttrOfType<ArrayAttr>("of");
    if (!tangentOf) {
      tangentOf = rewriter.getArrayAttr({});
    }
    LAGradContext lagradctx{moduleOp};
    // DEBUGpopulateFunc(lagradctx.debug_names, tangentFunc);
    // runActivityAnalysis(lagradctx, tangentFunc, tangentOf);
    if (failed(generateTangent(tangentFunc, lagradctx, tangentOf, rewriter))) {
      return nullptr;
    }
    return tangentFunc;
  }
};
} // namespace

void mlir::lagrad::populateLAGradTransforms(RewritePatternSet &patterns,
                                            MLIRContext *ctx) {
  patterns.add<ForwardModeAD>(ctx);
}
