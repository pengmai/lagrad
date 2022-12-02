/**
 * A custom pass meant to fill gaps in bufferizing tensor ops.
 * Motivated by a lack of bufferization for the tensor.insert op.
 */
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "LAGrad/Analysis.h"
#include "LAGrad/Logger.h"
#include "LAGrad/Passes.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using llvm::errs;
constexpr bool force_copy = false;

namespace {

void bufferizeLinalgOp(linalg::LinalgOp linalgOp, ValueRange outputBuffers,
                       ValueRange results, TypeConverter &typeConverter,
                       ConversionPatternRewriter &rewriter) {
  PatternRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(linalgOp);

  SmallVector<Value> newOperands;
  newOperands.reserve(linalgOp->getNumOperands());
  for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
    newOperands.push_back(rewriter.create<bufferization::ToMemrefOp>(
        linalgOp.getLoc(),
        typeConverter.convertType(inputOperand->get().getType()),
        inputOperand->get()));
  }
  newOperands.append(outputBuffers.begin(), outputBuffers.end());

  assert(linalgOp->getNumRegions() == 1 &&
         "expected linalg op to have 1 region");
  auto newOp = cast<linalg::LinalgOp>(linalgOp.cloneWithoutRegions());
  rewriter.inlineRegionBefore(linalgOp->getRegion(0), newOp->getRegion(0),
                              newOp->getRegion(0).begin());

  // The number of output buffers should always be 1 for now.
  rewriter.replaceOp(linalgOp, results);
}

class BufferizeInsertExtractPair
    : public OpConversionPattern<tensor::ExtractSliceOp> {
public:
  BufferizeInsertExtractPair(const InsertExtractAnalysis &ieAnalysis,
                             TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/3),
        ieAnalysis{ieAnalysis} {}
  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp extractSliceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!ieAnalysis.isPairedExtractSlice(extractSliceOp)) {
      return failure();
    }
    auto insertSliceOp = ieAnalysis.getPairedInsertSlice(extractSliceOp);
    auto sliceType = extractSliceOp.getType();
    auto sourceType = getTypeConverter()
                          ->convertType(extractSliceOp.getSourceType())
                          .cast<MemRefType>();
    Location loc = extractSliceOp.getLoc();

    // TODO: Need to see if we need to manually make this layout dynamic.
    auto resultType =
        memref::SubViewOp::inferRankReducedResultType(
            sliceType.getShape(), sourceType, extractSliceOp.getMixedOffsets(),
            extractSliceOp.getMixedSizes(), extractSliceOp.getMixedStrides())
            .cast<MemRefType>();
    auto identityResultType =
        getTypeConverter()->convertType(sliceType).cast<MemRefType>();
    Value source = rewriter.create<bufferization::ToMemrefOp>(
        loc, sourceType, extractSliceOp.getSource());
    Value subview = rewriter.create<memref::SubViewOp>(
        loc, resultType, source, extractSliceOp.getMixedOffsets(),
        extractSliceOp.getMixedSizes(), extractSliceOp.getMixedStrides());
    auto casted =
        rewriter.create<memref::CastOp>(loc, identityResultType, subview);
    auto loaded = rewriter.create<bufferization::ToTensorOp>(loc, casted);
    rewriter.replaceOp(extractSliceOp, loaded.getResult());
    rewriter.replaceOp(insertSliceOp, extractSliceOp.getSource());

    // Need to ensure linalg ops that write to the extractSliceOp are bufferized
    // in-place.
    for (auto &use : extractSliceOp.getResult().getUses()) {
      if (ieAnalysis.isLinalgMarkedForBufferization(use.getOwner())) {
        auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner());
        if (linalgOp.isDpsInit(&use)) {
          bufferizeLinalgOp(linalgOp, /*outputBuffers=*/subview,
                            /*results=*/loaded.getResult(), *getTypeConverter(),
                            rewriter);
        }
      }
    }
    return success();
  }

private:
  const InsertExtractAnalysis &ieAnalysis;
};

class BufferizeInsertOp : public OpConversionPattern<tensor::InsertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bufferization::ToTensorOp loaded;

    // This is very naïve.
    DominanceInfo dom;
    bool hasUseAfter = false;

    for (auto user : op.getDest().getUsers()) {
      if (dom.properlyDominates(op.getResult(), user)) {
        hasUseAfter = true;
      }
    }
    // constexpr bool bufferize_insert_in_place = false;
    // if (bufferize_insert_in_place) {
    if (!hasUseAfter && !force_copy) {
      // This first implementation updates the tensor in place rather than
      // returning a copy per the spec. Watch out for bugs this may cause.
      rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.getScalar(),
                                       adaptor.getDest(), adaptor.getIndices());
      loaded = rewriter.create<bufferization::ToTensorOp>(op.getLoc(),
                                                          adaptor.getDest());
    } else {
      Value space = rewriter.create<memref::AllocOp>(
          op.getLoc(), adaptor.getDest().getType().cast<MemRefType>());
      rewriter.create<linalg::CopyOp>(op.getLoc(), adaptor.getDest(), space);
      rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.getScalar(), space,
                                       adaptor.getIndices());
      loaded = rewriter.create<bufferization::ToTensorOp>(op.getLoc(), space);
    }

    op.replaceAllUsesWith(loaded.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * Replace tensor.extract_slice ops with a combination of memref.subview and
 * memref.cast.
 */
class BufferizeExtractSliceOp
    : public OpConversionPattern<tensor::ExtractSliceOp> {
private:
  const InsertExtractAnalysis &ieAnalysis;

public:
  using OpConversionPattern::OpConversionPattern;
  BufferizeExtractSliceOp(const InsertExtractAnalysis &ieAnalysis,
                          TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1),
        ieAnalysis{ieAnalysis} {}
  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getDroppedDims().empty() || ieAnalysis.isPairedExtractSlice(op)) {
      // Only deal with rank reduced cases
      // If in the ieAnalysis, the dedicated pass will deal with it.
      return failure();
    }
    auto resultTensorType =
        op.getResult().getType().dyn_cast_or_null<RankedTensorType>();
    // Curently only deal with 2D -> 1D and 3D -> 2D rank reduction cases.
    if (!resultTensorType ||
        op.getOffsetSizeAndStrideStartOperandIndex() != 1) {
      return failure();
    }
    auto sourceType = adaptor.getSource().getType().cast<MemRefType>();
    Value source = adaptor.getSource();
    // Ugly, brittle hack to get extract_slice with compressed GMMs working.
    // This results in the source having a fully dynamic layout map, which is
    // okay, but partial bufferization appears to expect an identity layout
    // map at the end of this transformation.
    if (sourceType.getDimSize(sourceType.getRank() - 1) == 1) {
      // removed make strided layout dynamic
      // source = rewriter.create<memref::CastOp>(op.getLoc(), sourceType, source);
    }

    auto resultType =
        memref::SubViewOp::inferRankReducedResultType(
            resultTensorType.getShape(), sourceType, op.getMixedOffsets(),
            op.getMixedSizes(), op.getMixedStrides())
            .cast<MemRefType>();
    auto identityResultType =
        getTypeConverter()->convertType(resultTensorType).cast<MemRefType>();

    Value subview = rewriter.create<memref::SubViewOp>(
        op.getLoc(), resultType, source, op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());

    DominanceInfo dom;
    bool hasWriteAfter = false;

    for (auto user : op.getSource().getUsers()) {
      if (dom.properlyDominates(op.getResult(), user)) {
        hasWriteAfter = true;
      }
    }

    // Need to consider writes to tensors that alias this one.
    // This causes a slow-down with GMM/main term, have yet to look into why.

    // This is a coarse-grained heuristic.
    // SmallVector<Value> frontier{op.source()};
    // ValueSet derivedFromSource;
    // runTopDownDFS(frontier, derivedFromSource);
    // for (Value derivedValue : derivedFromSource) {
    //   Operation *definingOp = derivedValue.getDefiningOp();
    //   if (definingOp && dom.properlyDominates(op.getResult(), definingOp) &&
    //       ((dyn_cast<tensor::InsertSliceOp>(definingOp) ||
    //         dyn_cast<tensor::InsertOp>(definingOp)))) {
    //     hasWriteAfter = true;
    //   }
    // }
    if (!hasWriteAfter && !force_copy) {
      // rewriter.replaceOpWithNewOp<memref::TensorLoadOp>(op,
      //                                                   subview.getResult());
      // rewriter.replaceOpWithNewOp<memref::CastOp>(op, identityResultType,
      //                                             subview);
      rewriter.replaceOp(op, subview);
    } else {
      Value dest =
          rewriter.create<memref::AllocOp>(op.getLoc(), identityResultType);
      rewriter.create<linalg::CopyOp>(op.getLoc(), subview, dest);
      rewriter.replaceOp(op, dest);
    }
    return success();
  }
};

class BufferizeInsertSliceOp
    : public OpConversionPattern<tensor::InsertSliceOp> {
private:
  const InsertExtractAnalysis &ieAnalysis;

public:
  using OpConversionPattern::OpConversionPattern;
  BufferizeInsertSliceOp(const InsertExtractAnalysis &ieAnalysis,
                         TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1),
        ieAnalysis{ieAnalysis} {}
  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (ieAnalysis.isPairedInsertSlice(op)) {
      // This op will be dealt with in the dedicated extract/insert slice pair
      // pass.
      return failure();
    }

    DominanceInfo dom;
    bool hasUseAfter = false;

    for (auto user : op.getDest().getUsers()) {
      if (dom.properlyDominates(op.getResult(), user)) {
        hasUseAfter = true;
      }
    }

    auto sliceType =
        memref::SubViewOp::inferRankReducedResultType(
            op.getSourceType().getShape(),
            adaptor.getDest().getType().cast<MemRefType>(),
            op.getMixedOffsets(), op.getMixedSizes(), op.getMixedStrides())
            .cast<MemRefType>();
    if (!hasUseAfter) {
      Value subview = rewriter.create<memref::SubViewOp>(
          op.getLoc(), sliceType, adaptor.getDest(), op.getMixedOffsets(),
          op.getMixedSizes(), op.getMixedStrides());
      rewriter.create<linalg::CopyOp>(op.getLoc(), adaptor.getSource(),
                                      subview);
      rewriter.replaceOp(op, adaptor.getDest());
    } else {
      Value dest = rewriter.create<memref::AllocOp>(
          op.getLoc(), adaptor.getDest().getType().dyn_cast<MemRefType>());
      rewriter.create<linalg::CopyOp>(op.getLoc(), adaptor.getDest(), dest);
      Value subview = rewriter.create<memref::SubViewOp>(
          op.getLoc(), sliceType, dest, op.getMixedOffsets(),
          op.getMixedSizes(), op.getMixedStrides());

      rewriter.create<linalg::CopyOp>(op.getLoc(), adaptor.getSource(),
                                      subview);
      rewriter.replaceOp(op, dest);
    }
    return success();
  }
};

// I might not need this so long as I run standalone-bufferize before tensor-bufferize.
class BufferizeForOp : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ForOp forOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (forOp.getNumIterOperands() == 0) {
      return failure();
    }
    Block &block = forOp.getRegion().front();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());
    SmallVector<bool, 4> keepMask;
    keepMask.reserve(yieldOp.getNumOperands());
    SmallVector<Value, 4> newBlockTransferArgs, newIterArgs, newYieldValues,
        newResultValues;
    newBlockTransferArgs.reserve(1 + forOp.getNumIterOperands());
    newBlockTransferArgs.push_back(Value()); // iv placeholder with null value
    newIterArgs.reserve(forOp.getNumIterOperands());
    newYieldValues.reserve(yieldOp.getNumOperands());
    newResultValues.reserve(forOp.getNumResults());
    BlockAndValueMapping oldToNew;
    for (auto it : llvm::zip(adaptor.getInitArgs(),     // iter from outside
                             forOp.getRegionIterArgs(), // iter inside region
                             forOp.getResults(),        // op results
                             yieldOp.getOperands()      // iter yield
                             )) {
      // auto result = std::get<2>(it);
      auto regionArg = std::get<1>(it);
      bool forwarded = regionArg.getType().isa<RankedTensorType>();
      keepMask.push_back(!forwarded);
      if (forwarded) {
        newBlockTransferArgs.push_back(std::get<0>(it));
        newResultValues.push_back(std::get<0>(it));
        Value loaded = rewriter.create<bufferization::ToTensorOp>(
            forOp.getLoc(), std::get<0>(it));
        regionArg.replaceAllUsesWith(loaded);
        continue;
      }
      newIterArgs.push_back(std::get<0>(it));
      newYieldValues.push_back(std::get<3>(it));
      newBlockTransferArgs.push_back(Value()); // placeholder with null value
      newResultValues.push_back(Value());      // placeholder with null value
    }

    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newIterArgs);
    Block &newBlock = newForOp.getRegion().front();

    // Replace the null placeholders with newly constructed values.
    newBlockTransferArgs[0] = newBlock.getArgument(0); // iv
    for (unsigned idx = 0, collapsedIdx = 0, e = newResultValues.size();
         idx != e; ++idx) {
      Value &blockTransferArg = newBlockTransferArgs[1 + idx];
      Value &newResultVal = newResultValues[idx];
      assert((blockTransferArg && newResultVal) ||
             (!blockTransferArg && !newResultVal));
      if (!blockTransferArg) {
        blockTransferArg = newForOp.getRegionIterArgs()[collapsedIdx];
        newResultVal = newForOp.getResult(collapsedIdx++);
      }
    }
    Block &oldBlock = forOp.getRegion().front();
    assert(oldBlock.getNumArguments() == newBlockTransferArgs.size() &&
           "unexpected argument size mismatch");
    // No results case: the scf::ForOp builder already created a zero
    // result terminator. Merge before this terminator and just get rid of the
    // original terminator that has been merged in.
    if (newIterArgs.empty()) {
      auto newYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
      rewriter.mergeBlockBefore(&oldBlock, newYieldOp, newBlockTransferArgs);
      rewriter.eraseOp(newBlock.getTerminator()->getPrevNode());
      rewriter.replaceOp(forOp, newResultValues);
      return success();
    }
    // No terminator case: merge and rewrite the merged terminator.
    auto cloneFilteredTerminator = [&](scf::YieldOp mergedTerminator) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(mergedTerminator);
      SmallVector<Value, 4> filteredOperands;
      filteredOperands.reserve(newResultValues.size());
      for (unsigned idx = 0, e = keepMask.size(); idx < e; ++idx)
        if (keepMask[idx])
          filteredOperands.push_back(mergedTerminator.getOperand(idx));
      rewriter.create<scf::YieldOp>(mergedTerminator.getLoc(),
                                    filteredOperands);
    };

    rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);
    auto mergedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
    cloneFilteredTerminator(mergedYieldOp);
    rewriter.eraseOp(mergedYieldOp);
    rewriter.replaceOp(forOp, newResultValues);
    return success();
  }
};

class BufferizeLinalgOp
    : public OpInterfaceConversionPattern<linalg::LinalgOp> {
public:
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;
  LogicalResult
  matchAndRewrite(linalg::LinalgOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.hasBufferSemantics()) {
      return failure();
    }
    DominanceInfo dom;
    bool hasUseAfter = false;
    for (auto user : op.getDpsInitOperand(0)->get().getUsers()) {
      if (dom.properlyDominates(op->getResult(0), user)) {
        hasUseAfter = true;
      }
    }
    SmallVector<Value> newOperands{operands.begin(), operands.end()};
    Operation *parentOp =
        op.getDpsInitOperand(0)->get().getDefiningOp()
            ?: op.getDpsInitOperand(0)->get().getParentRegion()->getParentOp();
    bool isImmutableArg = false;
    if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {
      auto blockArg = op.getDpsInitOperand(0)->get().cast<BlockArgument>();
      // Function arguments are immutable by default.
      isImmutableArg = !static_cast<bool>(
          funcOp.getArgAttr(blockArg.getArgNumber(), "linalg.mutable"));
    }
    bool isConstantMemory = isa_and_nonnull<arith::ConstantOp>(parentOp);
    bool mustCopy = hasUseAfter || isConstantMemory || isImmutableArg;

    assert(op.getNumDpsInits() == 1);
    if (mustCopy) {
      errs() << "Analysis determined we must copy: hasUseAfter " << hasUseAfter
             << " isConstantMemory: " << isConstantMemory
             << " isImmutableArg: " << isImmutableArg << "\n";
      Value space = rewriter.create<memref::AllocOp>(
          op.getLoc(), newOperands.back().getType().cast<MemRefType>());
      rewriter.create<linalg::CopyOp>(op.getLoc(), newOperands.back(), space);
      newOperands[newOperands.size() - 1] = space;
    } else {
      op.emitRemark() << "Made in-place update";
    }
    SmallVector<Value> results;
    results.reserve(op->getNumResults());
    Operation *unknownOp = op;
    auto linalgOp = cast<linalg::LinalgOp>(unknownOp);
    assert(op->getNumRegions() == 1 && "expected linalg op to have 1 region");
    auto newOp = cast<linalg::LinalgOp>(linalgOp.cloneWithoutRegions());
    rewriter.inlineRegionBefore(op->getRegion(0), newOp->getRegion(0),
                                newOp->getRegion(0).begin());
    for (auto outputBuffer : newOp.getDpsInitOperands()) {
      results.push_back(rewriter.create<bufferization::ToTensorOp>(
          op.getLoc(), outputBuffer->get()));
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

struct StandaloneBufferizePass
    : public PassWrapper<StandaloneBufferizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StandaloneBufferizePass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, linalg::LinalgDialect,
                    bufferization::BufferizationDialect>();
  }
  StringRef getArgument() const override { return "standalone-bufferize"; }
  StringRef getDescription() const override {
    return "Bufferize tensor ops required by the standalone dialect that "
           "aren't bufferized elsewhere.";
  }
  void runOnOperation() final {
    auto *context = &getContext();
    bufferization::BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    auto &ieAnalysis = getAnalysis<InsertExtractAnalysis>();

    target.addLegalDialect<memref::MemRefDialect>();
    target.addDynamicallyLegalDialect<arith::ArithDialect>([&](Operation *op) {
      return typeConverter.isLegal(op) || isa<arith::ConstantOp>(op);
    });
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
        [&](Operation *op) {
          return typeConverter.isLegal(op) ||
                 !ieAnalysis.isLinalgMarkedForBufferization(op);
        });
    target.addLegalOp<func::CallOp>();
    target.addLegalOp<func::ReturnOp>();
    target.addLegalOp<linalg::CopyOp>();
    target.addLegalOp<linalg::YieldOp>();
    // target.addLegalOp<linalg::InitTensorOp>();
    target.addIllegalOp<tensor::InsertOp>();
    target.addLegalDialect<scf::SCFDialect>();
    // target.addDynamicallyLegalOp<scf::ForOp>(
    //     [&](scf::ForOp op) { return typeConverter.isLegal(op); });
    bufferization::populateBufferizeMaterializationLegality(target);

    patterns.add<BufferizeInsertOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeInsertExtractPair>(ieAnalysis, typeConverter,
                                             patterns.getContext());
    patterns.add<BufferizeExtractSliceOp>(ieAnalysis, typeConverter,
                                          patterns.getContext());
    patterns.add<BufferizeInsertSliceOp>(ieAnalysis, typeConverter,
                                         patterns.getContext());
    // patterns.add<BufferizeForOp>(typeConverter, patterns.getContext());
    // patterns.add<BufferizeLinalgOp>(typeConverter,
    // patterns.getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {

      signalPassFailure();
    }
  };
};
} // namespace

std::unique_ptr<Pass> mlir::lagrad::createBufferizePass() {
  return std::make_unique<StandaloneBufferizePass>();
}
