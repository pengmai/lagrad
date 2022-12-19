/**
 * MLIR's canonicalization will handle a lot of basic DCE, but cleaning unused
 * iteration arguments of scf.for ops will unfortunately persist.
 */
#include "LAGrad/Analysis.h"
#include "LAGrad/Logger.h"
#include "LAGrad/Passes.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using llvm::errs;

namespace {
class RemoveUnusedInputs : public OpRewritePattern<linalg::GenericOp> {
public:
  RemoveUnusedInputs(MLIRContext *ctx)
      : OpRewritePattern<linalg::GenericOp>(ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    auto regionArgs = genericOp.getBodyRegion().getArguments();
    auto regionInputArgs = regionArgs.take_front(genericOp.getNumInputs());
    bool canonicalize = false;
    SmallVector<Value> new_inputs;
    new_inputs.reserve(genericOp.getNumInputs());
    SmallVector<Value> old_region_operands;
    old_region_operands.reserve(genericOp.getNumInputs());
    SmallVector<AffineMap> new_indexing_maps;
    auto indexing_maps = genericOp.getIndexingMaps();
    new_indexing_maps.reserve(genericOp.getNumInputsAndOutputs());
    for (size_t i = 0; i < regionInputArgs.size(); i++) {
      auto regionArg = regionInputArgs[i];
      bool unused = regionArg.use_empty();
      canonicalize |= unused;

      if (!unused) {
        new_inputs.push_back(genericOp.getInputOperand(i)->get());
        new_indexing_maps.push_back(indexing_maps[i]);
        old_region_operands.push_back(regionArg);
      }
    }
    if (!canonicalize) {
      return failure();
    }

    // Copy over the output indexing maps
    for (int64_t i = genericOp.getNumInputs();
         i < genericOp.getNumInputsAndOutputs(); i++) {
      new_indexing_maps.push_back(indexing_maps[i]);
    }
    for (auto output_arg : genericOp.getRegionOutputArgs()) {
      old_region_operands.push_back(output_arg);
    }

    SmallVector<Value> outputs;
    outputs.reserve(genericOp.getNumOutputs());
    for (auto outputOperand : genericOp.getOutputOperands()) {
      outputs.push_back(outputOperand->get());
    }

    SmallVector<llvm::StringRef> iterator_types{
        genericOp.iterator_types().getAsValueRange<StringAttr>()};

    SmallVector<Value> genericOperands;
    for (Value arg : genericOp.getBodyRegion().getArguments()) {
      genericOperands.push_back(arg);
    }
    auto newOp = rewriter.create<linalg::GenericOp>(
        genericOp.getLoc(), /*resultTensorType=*/genericOp.getResultTypes(),
        new_inputs, outputs, new_indexing_maps, iterator_types,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          assert(regionArgs.size() == old_region_operands.size() &&
                 "Unexpected size mismatch");
          cloneBasicBlock(genericOp.getBodyRegion().getOps(), builder,
                          regionArgs, old_region_operands,
                          /*offsetInputs=*/false);
        });

    rewriter.replaceOp(genericOp, newOp.getResults());
    return success();
  }
};

bool isZeroTensor(Value value) {
  if (auto constantOp =
          dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp())) {
    if (auto splatAttr =
            constantOp.getValueAttr().dyn_cast<SplatElementsAttr>()) {
      if (auto floatAttr = splatAttr.getSplatValue<FloatAttr>()) {
        return !floatAttr.getValue().isNonZero();
      }
    }
  }
  return false;
}

bool isConstantOnesTensor(Value value) {
  if (auto fillOp = dyn_cast_or_null<linalg::FillOp>(value.getDefiningOp())) {
    if (auto constantFloatOp = dyn_cast_or_null<arith::ConstantFloatOp>(
            fillOp.value().getDefiningOp())) {
      if (constantFloatOp.value().convertToDouble() == 1.0) {
        return true;
      }
    }
  }
  return false;
}

class FoldMultiplyByOnes : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasTensorSemantics()) {
      return failure();
    }
    if (llvm::none_of(op.getOperands(),
                      [](Value v) { return isConstantOnesTensor(v); })) {
      return failure();
    }
    size_t idx = 0;
    for (OpOperand *operand : op.getInputTensorOperands()) {
      if (isConstantOnesTensor(operand->get()) &&
          op.payloadUsesValueFromOperand(operand)) {
        BlockArgument regionArg = op.getBody()->getArgument(idx);
        for (Operation *user : regionArg.getUsers()) {
          if (auto mulfOp = dyn_cast<arith::MulFOp>(user)) {
            Value other = mulfOp.getLhs() == regionArg ? mulfOp.getRhs()
                                                       : mulfOp.getLhs();
            rewriter.updateRootInPlace(
                op, [&]() { rewriter.replaceOp(mulfOp, other); });
            return success();
          }
        }
      }
      idx++;
    }
    return failure();
  }
};

class FoldAddToZero : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasTensorSemantics()) {
      return failure();
    }
    bool isOutputZero =
        op.getNumResults() == 1 && isZeroTensor(op.getOutputOperand(0)->get());
    if (!(isOutputZero && op.getNumInputs() == 1 &&
          op.getOperandTypes()[0].cast<RankedTensorType>() ==
              op.getOutputTensorTypes()[0])) {
      return failure();
    }

    auto yieldOp = cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (auto addfOp = dyn_cast_or_null<arith::AddFOp>(
            yieldOp.getOperand(0).getDefiningOp())) {
      BlockArgument inputArg = op.getBody()->getArgument(0);
      BlockArgument outputArg = op.getBody()->getArgument(1);
      if ((addfOp.getLhs() == inputArg && addfOp.getRhs() == outputArg) ||
          (addfOp.getLhs() == outputArg && addfOp.getRhs() == inputArg)) {
        errs() << "Replaced addtozero: " << op << "\n";
        rewriter.replaceOp(op, op.getInputOperand(0)->get());
        return success();
      }
    }
    return failure();
  }
};

class FusePairedInsertExtract : public RewritePattern {
private:
  const InsertExtractAnalysis &ieAnalysis;

public:
  FusePairedInsertExtract(const InsertExtractAnalysis &ieAnalysis,
                          MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        ieAnalysis(ieAnalysis) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op);
    if (!(extractSliceOp && ieAnalysis.isPairedExtractSlice(extractSliceOp))) {
      return failure();
    }

    DominanceInfo dom;

    for (Operation *user : extractSliceOp.getResult().getUsers()) {
      for (Value operand : user->getOperands()) {
        if (operand != extractSliceOp.getResult()) {
          if (auto linalgOp =
                  dyn_cast_or_null<linalg::LinalgOp>(operand.getDefiningOp())) {
            assert(linalgOp.getNumOutputs() == 1 &&
                   "expected linalg op to have 1 output");
            if (linalgOp.hasTensorSemantics() &&
                dom.dominates(extractSliceOp.source(), linalgOp) &&
                isZeroTensor(linalgOp.getOutputTensorOperands()[0]->get())) {
              rewriter.setInsertionPoint(linalgOp);
              Location loc = op->getLoc();
              auto pairedInsertSlice =
                  ieAnalysis.getPairedInsertSlice(extractSliceOp);
              auto movedExtractSlice = rewriter.create<tensor::ExtractSliceOp>(
                  loc, extractSliceOp.getType(), extractSliceOp.source(),
                  extractSliceOp.getMixedOffsets(),
                  extractSliceOp.getMixedSizes(),
                  extractSliceOp.getMixedStrides());

              SmallVector<Value> newOperands;
              newOperands.reserve(linalgOp.getNumInputsAndOutputs());
              for (OpOperand *opOperand : linalgOp.getInputTensorOperands()) {
                newOperands.push_back(opOperand->get());
              }
              for (OpOperand *opOperand : linalgOp.getOutputTensorOperands()) {
                if (isZeroTensor(opOperand->get())) {
                  newOperands.push_back(movedExtractSlice.getResult());
                } else {
                  llvm_unreachable("not yet implemented");
                }
              }
              auto clonedOp = linalgOp.clone(
                  rewriter, loc, linalgOp->getResultTypes(), newOperands);
              rewriter.replaceOp(extractSliceOp, movedExtractSlice.getResult());
              rewriter.replaceOp(linalgOp, clonedOp->getResults());
              // Creating a new insert slice op segfaults for some reason.
              pairedInsertSlice.sourceMutable().assign(clonedOp->getResult(0));
              rewriter.eraseOp(user);
              // Logger::blue("Ran fuse paired insert extract");
              return success();
            }
          }
        }
      }
    }
    return failure();
  }
};
} // namespace

namespace {
struct LinalgCanonicalizePass
    : public PassWrapper<LinalgCanonicalizePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  StringRef getArgument() const override { return "linalg-canonicalize"; }
  StringRef getDescription() const override {
    return "Additional custom canonicalization passes for the linalg "
           "dialect.";
  }
  void runOnOperation() final {
    auto *context = &getContext();
    auto &ieAnalysis = getAnalysis<InsertExtractAnalysis>();
    RewritePatternSet patterns(context);
    patterns.add<RemoveUnusedInputs>(patterns.getContext());
    // patterns.add<FoldMultiplyByOnes>(patterns.getContext());
    // patterns.add<FoldAddToZero>(patterns.getContext());
    patterns.add<FusePairedInsertExtract>(ieAnalysis, patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::lagrad::createLinalgCanonicalizePass() {
  return std::make_unique<LinalgCanonicalizePass>();
}