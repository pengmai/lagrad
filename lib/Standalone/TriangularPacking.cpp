#include "Standalone/Logger.h"
#include "Standalone/Passes.h"
#include "Standalone/StandaloneOps.h"
#include "Standalone/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using llvm::errs;

namespace {
bool hasPackedEncoding(Type type) {
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
    if (auto encoding =
            rankedTensorType.getEncoding().dyn_cast_or_null<StringAttr>()) {
      if (encoding.getValue() == "pltri") {
        return true;
      }
    }
  }
  return false;
}

bool hasPackedEncoding(linalg::LinalgOp op) {
  if (!op.hasTensorSemantics()) {
    return false;
  }
  for (OpOperand *operand : op.getInputAndOutputOperands()) {
    if (hasPackedEncoding(operand->get().getType())) {
      return true;
    }
  }
  return false;
}

RankedTensorType convertToPackedType(RankedTensorType tensorType) {
  if (tensorType.isDynamicDim(tensorType.getRank() - 1)) {
    llvm_unreachable("dynamic packed triangular matrices not yet implemented");
  }

  int64_t triDim = tensorType.getShape().back();
  assert(triDim == tensorType.getShape().rbegin()[1] &&
         "triangular packed shape was not square");
  int64_t triSize = triDim * (triDim - 1) / 2;
  SmallVector<int64_t> packedShape =
      llvm::to_vector<4>(tensorType.getShape().drop_back(1));
  packedShape.back() = triSize;
  return RankedTensorType::get(packedShape, tensorType.getElementType());
}

/// Taken from mlir/Dialect/Linalg/Utils/Utils.cpp
/// Given a list of subview ranges, extract individual values for lower, upper
/// bounds and steps and put them into the corresponding vectors.
static void unpackRanges(ArrayRef<Range> ranges, SmallVectorImpl<Value> &lbs,
                         SmallVectorImpl<Value> &ubs,
                         SmallVectorImpl<Value> &steps) {
  for (Range range : ranges) {
    lbs.emplace_back(range.offset);
    ubs.emplace_back(range.size);
    steps.emplace_back(range.stride);
  }
}

static SmallVector<Value> makeCanonicalAffineApplies(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ArrayRef<Value> vals,
                                                     Type operandType,
                                                     ValueRange trilIndices) {
  if (map.isEmpty())
    return {};

  assert(map.getNumInputs() == vals.size());
  SmallVector<Value> res;
  res.reserve(map.getNumResults());
  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    if (hasPackedEncoding(operandType)) {
      auto exprMap = AffineMap::get(dims, map.getNumSymbols(), e);
      res.push_back(b.create<AffineApplyOp>(loc, exprMap, trilIndices));
    } else {
      auto exprMap = AffineMap::get(dims, map.getNumSymbols(), e);
      SmallVector<Value> operands(vals.begin(), vals.end());
      canonicalizeMapAndOperands(&exprMap, &operands);
      res.push_back(b.create<AffineApplyOp>(loc, exprMap, operands));
    }
  }
  return res;
}

// Modified from "mlir/Dialect/Linalg/Utils/Utils.cpp"
static SmallVector<Value>
inlineRegionAndEmitStore(OpBuilder &b, Location loc, linalg::LinalgOp op,
                         ArrayRef<Value> indexedValues,
                         ArrayRef<SmallVector<Value>> indexing,
                         ArrayRef<Value> outputTensors) {
  auto &block = op->getRegion(0).front();
  BlockAndValueMapping map;
  map.map(block.getArguments(), indexedValues);
  for (auto &op : block.without_terminator()) {
    auto *newOp = b.clone(op, map);
    map.map(op.getResults(), newOp->getResults());
  }

  Operation *terminator = block.getTerminator();
  SmallVector<Value> results;
  for (OpOperand &operand : terminator->getOpOperands()) {
    Value toStore = map.lookupOrDefault(operand.get());
    results.push_back(b.create<tensor::InsertOp>(
        loc, toStore, outputTensors[operand.getOperandNumber()],
        indexing[operand.getOperandNumber()]));
  }
  return results;
}

static SmallVector<Value> emitScalarImplementation(OpBuilder &b, Location loc,
                                                   ValueRange allIvs,
                                                   ValueRange iterArgs,
                                                   linalg::LinalgOp linalgOp,
                                                   Value triDim) {
  assert(iterArgs.size() == linalgOp.getOutputTensorOperands().size() &&
         "Expected # of iter args to be equal to # of output tensor operands.");

  SmallVector<Value> indexedValues;
  indexedValues.reserve(linalgOp.getNumInputsAndOutputs());

  auto allIvsPlusDims = SmallVector<Value>(allIvs.begin(), allIvs.end());

  // Emit the required indexing. Assume for now it's packed (strictly) lower
  // triangular. Might be a faster way to compute this.
  /* j - (i + 1) + i * (2 * d - (i + 1)) / 2; */
  auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
  auto one = b.create<arith::ConstantIndexOp>(loc, 1);
  auto two = b.create<arith::ConstantIndexOp>(loc, 2);
  auto iv = allIvsPlusDims.rbegin()[1];
  auto ivPlusOne = b.create<arith::AddIOp>(loc, iv, one);
  auto jv = allIvsPlusDims.back();
  auto LidxInit = b.create<arith::DivUIOp>(
      loc,
      b.create<arith::MulIOp>(
          loc, iv,
          b.create<arith::SubIOp>(
              loc, b.create<arith::MulIOp>(loc, two, triDim), ivPlusOne)),
      two);
  auto Lidx = b.create<arith::AddIOp>(
      loc, b.create<arith::SubIOp>(loc, jv, ivPlusOne), LidxInit);

  // The zero is a temporary placeholder to pass the
  // verifier. We expect it to be removed when the packed annotation is
  // converted to the proper packed type.
  auto trilIndices = ValueRange{zero, Lidx};

  // 1.a. Emit load from input operands or for scalars access the operand
  // itself.
  for (auto inputOperand : linalgOp.getInputOperands()) {
    if (linalgOp.isScalar(inputOperand)) {
      indexedValues.push_back(inputOperand->get());
      continue;
    }
    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getTiedIndexingMap(inputOperand), allIvsPlusDims,
        inputOperand->get().getType(), trilIndices);
    indexedValues.push_back(
        b.create<tensor::ExtractOp>(loc, inputOperand->get(), indexing));
  }

  // 1.b. Emit load from output views.
  for (auto pair : llvm::zip(linalgOp.getOutputOperands(), iterArgs)) {
    auto outputOperand = std::get<0>(pair);
    Value iterArg = std::get<1>(pair);

    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getTiedIndexingMap(outputOperand), allIvsPlusDims,
        outputOperand->get().getType(), trilIndices);
    indexedValues.push_back(
        b.create<tensor::ExtractOp>(loc, iterArg, indexing));
  }

  // 2. Inline region, currently only works for a single basic block.
  // 3. Emit store.
  SmallVector<Value> outputTensors{iterArgs};
  SmallVector<SmallVector<Value>, 8> indexing;
  for (auto outputOperand : linalgOp.getOutputTensorOperands()) {
    indexing.push_back(makeCanonicalAffineApplies(
        b, loc, linalgOp.getTiedIndexingMap(outputOperand), allIvsPlusDims,
        outputOperand->get().getType(), trilIndices));
  }
  return inlineRegionAndEmitStore(b, loc, linalgOp, indexedValues, indexing,
                                  outputTensors);
}

class PackLinalgOp : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasTensorSemantics()) {
      return failure();
    }
    if (!hasPackedEncoding(op)) {
      return failure();
    }

    auto loopRanges = op.createLoopRanges(rewriter, op.getLoc());
    auto iteratorTypes = llvm::to_vector<4>(op.iterator_types().getValue());
    SmallVector<Value, 4> lbs, ubs, steps;
    unpackRanges(loopRanges, lbs, ubs, steps);
    // Initialize space for outputs
    SmallVector<Value> iterArgsInit;
    iterArgsInit.reserve(op.getNumOutputs());
    for (OpOperand *outTensor : op.getOutputTensorOperands()) {
      // TODO: This is a bandaid, need some way of determining when it's safe to
      // write.
      if (isa_and_nonnull<tensor::ExtractSliceOp>(
              outTensor->get().getDefiningOp())) {
        iterArgsInit.push_back(outTensor->get());
      } else {
        auto outType = outTensor->get().getType().cast<RankedTensorType>();

        auto memrefType =
            hasPackedEncoding(outType)
                ? BufferizeTypeConverter().convertType(
                      convertToPackedType(outType))
                : MemRefType::get(outType.getShape(), outType.getElementType());
        Value space = rewriter.create<linalg::InitTensorOp>(
            op.getLoc(), outType.getShape(), outType.getElementType());
        if (hasPackedEncoding(outType)) {
          space =
              rewriter.create<standalone::PackOp>(op.getLoc(), outType, space);
        }
        if (op.payloadUsesValueFromOperand(outTensor)) {
          auto castedSpace = rewriter.create<memref::BufferCastOp>(
              op.getLoc(), memrefType, space);
          auto memrefOutput = rewriter.create<memref::BufferCastOp>(
              op.getLoc(), memrefType, outTensor->get());
          rewriter.create<linalg::CopyOp>(op.getLoc(), memrefOutput,
                                          castedSpace);
        }

        iterArgsInit.push_back(space);
      }
    }
    auto loopNest = scf::buildLoopNest(
        rewriter, op.getLoc(), lbs, ubs, steps, iterArgsInit,
        [&](OpBuilder &b, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> scf::ValueVector {
          auto iterNext =
              emitScalarImplementation(b, loc, ivs, iterArgs, op, ubs.back());
          return scf::ValueVector{iterNext.begin(), iterNext.end()};
        });

    // Modify loop bounds. This is only valid for the column major, strictly
    // lower triangular case.
    auto lastLoop = loopNest.loops.back();
    auto secondLastLoop = loopNest.loops.rbegin()[1];
    rewriter.setInsertionPoint(lastLoop);
    auto last_lb = rewriter.create<arith::AddIOp>(
        op.getLoc(), secondLastLoop.getInductionVar(),
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1));
    lastLoop.setLowerBound(last_lb);

    rewriter.replaceOp(op, loopNest.getResults());
    loopNest.loops.front()->setAttr("Packed loop",
                                    UnitAttr::get(rewriter.getContext()));
    return success();
  }
};

struct PackedTensorUsageAnalysis {
  PackedTensorUsageAnalysis(Operation *op) {
    op->walk([&](Operation *childOp) {
      auto packedPredicate = [&](Type type) { return hasPackedEncoding(type); };
      if (llvm::any_of(childOp->getOperandTypes(), packedPredicate) ||
          llvm::any_of(childOp->getResultTypes(), packedPredicate)) {
        cache.insert(childOp);
      }
    });
  }

  bool usesPackedTensor(Operation *op) const { return cache.contains(op); }
  void removeMark(Operation *op) { cache.erase(op); }

private:
  DenseSet<Operation *> cache;
};

class ErasePackedFuncOp : public OpRewritePattern<FuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const override {
    op.setType(stripEncodingFromFunc(op.getType()));
    rewriter.updateRootInPlace(op, [&]() {
      for (auto arg : op.getArguments()) {
        if (hasPackedEncoding(arg.getType())) {
          arg.setType(
              convertToPackedType(arg.getType().cast<RankedTensorType>()));
        }
      }
    });
    return success();
  }

private:
  FunctionType stripEncodingFromFunc(FunctionType funcTyp) const {
    SmallVector<Type> inputTypes;
    for (auto typ : funcTyp.getInputs()) {
      if (hasPackedEncoding(typ)) {
        inputTypes.push_back(convertToPackedType(typ.cast<RankedTensorType>()));
      } else {
        inputTypes.push_back(typ);
      }
    }
    SmallVector<Type> outputTypes;
    for (auto typ : funcTyp.getResults()) {
      if (hasPackedEncoding(typ)) {
        outputTypes.push_back(
            convertToPackedType(typ.cast<RankedTensorType>()));
      } else {
        outputTypes.push_back(typ);
      }
    }
    return FunctionType::get(funcTyp.getContext(), inputTypes, outputTypes);
  }
};

class ErasePackedCallOp : public OpRewritePattern<CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CallOp op,
                                PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto funcOp = cast<FuncOp>(moduleOp.lookupSymbol(op.calleeAttr()));
    rewriter.replaceOpWithNewOp<CallOp>(op, funcOp, op.operands());
    return success();
  }
};

class ErasePackedExtractOp : public OpRewritePattern<tensor::ExtractOp> {
private:
  const PackedTensorUsageAnalysis &packedTensorUsage;

public:
  ErasePackedExtractOp(const PackedTensorUsageAnalysis &packedTensorUsage,
                       MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/1), packedTensorUsage{
                                                      packedTensorUsage} {}

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    if (!packedTensorUsage.usesPackedTensor(op)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, op.tensor(),
                                                   op.indices().drop_back(1));
    return success();
  }
};

class ErasePackedInsertOp : public OpRewritePattern<tensor::InsertOp> {
private:
  const PackedTensorUsageAnalysis &packedTensorUsage;

public:
  ErasePackedInsertOp(const PackedTensorUsageAnalysis &packedTensorUsage,
                      MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/1), packedTensorUsage{
                                                      packedTensorUsage} {}
  LogicalResult matchAndRewrite(tensor::InsertOp op,
                                PatternRewriter &rewriter) const override {
    if (!packedTensorUsage.usesPackedTensor(op)) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, op.scalar(), op.dest(),
                                                  op.indices().drop_back(1));
    return success();
  }
};

class ErasePackedExtractSliceOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {
private:
  const PackedTensorUsageAnalysis &packedTensorUsage;

public:
  ErasePackedExtractSliceOp(const PackedTensorUsageAnalysis &packedTensorUsage,
                            MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/1), packedTensorUsage{
                                                      packedTensorUsage} {}
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {
    if (!packedTensorUsage.usesPackedTensor(op)) {
      return failure();
    }

    int64_t triDim = op.getType().getShape().back();
    assert(triDim == op.getType().getShape().rbegin()[1] &&
           "triangular packed shape was not square");
    int64_t triSize = triDim * (triDim - 1) / 2;
    SmallVector<OpFoldResult, 4> offsets{op.getMixedOffsets()},
        sizes{op.getMixedSizes()}, strides{op.getMixedStrides()};
    // TODO: rework this to be more general. It currently assumes the full
    // triangular matrix is extracted and the offset is always the same (i.e. is
    // zero).
    offsets.pop_back();
    sizes.pop_back();
    strides.pop_back();

    sizes.back() =
        IntegerAttr::get(IndexType::get(rewriter.getContext()), triSize);

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, convertToPackedType(op.getType()), op.source(), offsets, sizes,
        strides);
    return success();
  }
};

class ErasePackedInsertSliceOp
    : public OpRewritePattern<tensor::InsertSliceOp> {
private:
  const PackedTensorUsageAnalysis &packedTensorUsage;

public:
  ErasePackedInsertSliceOp(const PackedTensorUsageAnalysis &packedTensorUsage,
                           MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/1), packedTensorUsage{
                                                      packedTensorUsage} {}
  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    if (!packedTensorUsage.usesPackedTensor(op)) {
      return failure();
    }

    int64_t triDim = op.getType().getShape().back();
    assert(triDim == op.getType().getShape().rbegin()[1] &&
           "triangular packed shape was not square");
    int64_t triSize = triDim * (triDim - 1) / 2;
    SmallVector<OpFoldResult, 4> offsets{op.getMixedOffsets()},
        sizes{op.getMixedSizes()}, strides{op.getMixedStrides()};
    // TODO: rework this to be more general. It currently assumes the full
    // triangular matrix is extracted and the offset is always the same (i.e. is
    // zero).
    offsets.pop_back();
    sizes.pop_back();
    strides.pop_back();

    sizes.back() =
        IntegerAttr::get(IndexType::get(rewriter.getContext()), triSize);

    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, op.source(), op.dest(), offsets, sizes, strides);
    return success();
  }
};

class EraseCastOp : public OpRewritePattern<tensor::CastOp> {
  const PackedTensorUsageAnalysis &packedTensorUsage;

public:
  EraseCastOp(const PackedTensorUsageAnalysis &packedTensorUsage,
              MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/1), packedTensorUsage{
                                                      packedTensorUsage} {}
  LogicalResult matchAndRewrite(tensor::CastOp op,
                                PatternRewriter &rewriter) const override {
    if (!packedTensorUsage.usesPackedTensor(op)) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(), op.source());
    return success();
  }
};

class ErasePackOp : public OpRewritePattern<standalone::PackOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(standalone::PackOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasPackedEncoding(op.getType())) {
      return failure();
    }
    auto sourceType = op.source().getType().dyn_cast<RankedTensorType>();
    auto destType = op.getType().dyn_cast<RankedTensorType>();
    // Meant to handle a specific case where an explicit pack op is used because
    // linalg.init_tensor can't make tensors with special encodings.
    if (!(sourceType && destType &&
          sourceType.getShape() == destType.getShape() &&
          sourceType.getElementType() == destType.getElementType() &&
          isa_and_nonnull<linalg::InitTensorOp>(op.source().getDefiningOp()))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<linalg::InitTensorOp>(
        op, convertToPackedType(destType).getShape(),
        destType.getElementType());
    return success();
  }
};

class EraseEncoding : public RewritePattern {
public:
  EraseEncoding(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (isa<tensor::TensorDialect>(op->getDialect())) {
      return failure();
    }
    if (llvm::none_of(op->getOperandTypes(),
                      [&](Type type) { return hasPackedEncoding(type); }) &&
        llvm::none_of(op->getResultTypes(),
                      [&](Type type) { return hasPackedEncoding(type); })) {
      return failure();
    }

    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      if (auto valueAttr = constOp.valueAttr().dyn_cast<SplatElementsAttr>()) {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            constOp,
            DenseElementsAttr::get(
                convertToPackedType(constOp.getType().cast<RankedTensorType>()),
                valueAttr.getSplatValue()));
      } else {
        auto denseAttr = constOp.valueAttr().cast<DenseFPElementsAttr>();
        auto tensorType = denseAttr.getType().cast<RankedTensorType>();
        int64_t d = tensorType.getShape().back();
        SmallVector<APFloat> packedValues;
        packedValues.reserve(d * (d - 1) / 2);
        int64_t dimIdx = 1;
        for (int64_t dim : tensorType.getShape().drop_back(2)) {
          int64_t stride = 1;
          for (int64_t stride_dim : tensorType.getShape().drop_front(dimIdx)) {
            stride *= stride_dim;
          }
          for (int64_t m = 0; m < dim; m++) {
            for (int64_t i = 0; i < d; i++) {
              for (int64_t j = i + 1; j < d; j++) {
                packedValues.push_back(
                    denseAttr.getFlatValue<APFloat>(m * stride + j * d + i));
              }
            }
          }
          dimIdx++;
        }
        auto packedAttr = DenseFPElementsAttr::get(
            convertToPackedType(tensorType), packedValues);
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp, packedAttr);
      }
    } else {
      rewriter.updateRootInPlace(op, [&]() {
        for (auto operand : op->getOperands()) {
          if (hasPackedEncoding(operand.getType())) {
            operand.setType(convertToPackedType(
                operand.getType().cast<RankedTensorType>()));
          }
        }
        for (auto result : op->getResults()) {
          if (hasPackedEncoding(result.getType())) {
            result.setType(
                convertToPackedType(result.getType().cast<RankedTensorType>()));
          }
        }
      });
    }
    return success();
  }
};

class ErasePackedForOp : public OpRewritePattern<scf::ForOp> {
private:
public:
  ErasePackedForOp(MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Could remove this in favour of the more generic EraseEncoding
    // pattern, would need to extend it to traverse region arguments.
    rewriter.updateRootInPlace(op, [&]() {
      for (BlockArgument operand : op.getRegionIterArgs()) {
        if (hasPackedEncoding(operand.getType())) {
          operand.setType(
              convertToPackedType(operand.getType().cast<RankedTensorType>()));
        }
      }
    });
    return success();
  }
};

struct PackTriangularPass
    : public PassWrapper<PackTriangularPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, scf::SCFDialect>();
  }
  StringRef getArgument() const override { return "pack-triangular"; }
  StringRef getDescription() const override {
    return "Convert linalg ops that operate on triangular tensors to "
           "packed loops.";
  }
  void runOnOperation() final {
    auto *context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalOp<linalg::InitTensorOp>();
    target.addLegalOp<linalg::FillOp>();
    target.addLegalOp<linalg::CopyOp>();
    target.addLegalOp<linalg::YieldOp>();
    target.addLegalOp<standalone::PackOp>();

    patterns.add<PackLinalgOp>(patterns.getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    auto &packedTensorUsage = getAnalysis<PackedTensorUsageAnalysis>();
    RewritePatternSet erasurePatterns(context);
    ConversionTarget erasureTarget(*context);

    auto packedPredicate = [&](Operation *op) {
      return !packedTensorUsage.usesPackedTensor(op);
    };
    erasureTarget.addDynamicallyLegalDialect<tensor::TensorDialect>(
        packedPredicate);
    erasureTarget.addDynamicallyLegalOp<CallOp>(packedPredicate);
    erasureTarget.addDynamicallyLegalDialect<arith::ArithmeticDialect>(
        [&](Operation *op) {
          auto isPacked = [&](Type type) { return hasPackedEncoding(type); };
          return llvm::none_of(op->getOperandTypes(), isPacked) &&
                 llvm::none_of(op->getResultTypes(), isPacked);
        });
    erasureTarget.addDynamicallyLegalDialect<scf::SCFDialect>(
        [&](Operation *op) {
          auto isPacked = [&](Type type) { return hasPackedEncoding(type); };
          return llvm::none_of(op->getOperandTypes(), isPacked) &&
                 llvm::none_of(op->getResultTypes(), isPacked);
        });
    erasureTarget.addLegalOp<linalg::InitTensorOp>();
    erasureTarget.addLegalOp<linalg::CopyOp>();
    // erasureTarget.addLegalOp<linalg::FillOp>();
    // erasureTarget.addLegalOp<linalg::YieldOp>();

    erasureTarget.addIllegalDialect<standalone::StandaloneDialect>();
    erasurePatterns.add<ErasePackedFuncOp>(context);
    erasurePatterns.add<ErasePackedCallOp>(context);
    erasurePatterns.add<ErasePackedExtractOp>(packedTensorUsage, context);
    erasurePatterns.add<ErasePackedInsertOp>(packedTensorUsage, context);
    erasurePatterns.add<ErasePackedExtractSliceOp>(packedTensorUsage, context);
    erasurePatterns.add<ErasePackedInsertSliceOp>(packedTensorUsage, context);
    erasurePatterns.add<EraseCastOp>(packedTensorUsage, context);
    erasurePatterns.add<ErasePackOp>(context);
    erasurePatterns.add<EraseEncoding>(context);
    erasurePatterns.add<ErasePackedForOp>(context);
    if (failed(applyPartialConversion(getOperation(), erasureTarget,
                                      std::move(erasurePatterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::Standalone::createPackTriangularPass() {
  return std::make_unique<PackTriangularPass>();
}
