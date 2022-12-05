#include "LAGrad/Analysis.h"
#include "LAGrad/Passes.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

using llvm::errs;
namespace mlir {
inline raw_ostream &operator<<(raw_ostream &os, HotSparsityType type) {
  switch (type) {
  case HotSparsityType::Empty:
    os << "[empty]";
    break;
  case HotSparsityType::OneHot:
    os << "[one-hot]";
    break;
  case HotSparsityType::RowHot:
    os << "[row-hot]";
    break;
  case HotSparsityType::ColHot:
    os << "[col-hot]";
    break;
  }
  return os;
}

SparsePropagation::SparsePropagation(Operation *op, AnalysisManager &am) {
  // op->walk([&](FuncOp funcOp) { DEBUGpopulateFunc(debug_names, funcOp); });
  op->walk([&](FuncOp funcOp) {
    for (BlockArgument arg : funcOp.getArguments()) {
      if (auto rankedTensorType = arg.getType().dyn_cast<RankedTensorType>()) {
        auto sparsityEncoding = getSparsityEncoding(rankedTensorType);
        if (sparsityEncoding.hasValue()) {
          sparsityTypes[arg] = sparsityEncoding.getValue();
        }
      }
    }
  });

  op->walk([&](linalg::FillOp fillOp) {
    if (auto floatOp = dyn_cast_or_null<arith::ConstantFloatOp>(
            fillOp.value().getDefiningOp())) {
      if (floatOp.value().isZero()) {
        this->sparsityTypes[fillOp.getResult(0)] = HotSparsityType::Empty;
      }
    }
  });
  op->walk([&](arith::ConstantOp constOp) {
    if (auto valueAttr = constOp.valueAttr().dyn_cast<SplatElementsAttr>()) {
      if (auto splatAttr = valueAttr.getSplatValue().dyn_cast<FloatAttr>()) {
        if (splatAttr.getValue().isZero()) {
          sparsityTypes[constOp.getResult()] = HotSparsityType::Empty;
        }
      }
    }
  });
  auto &lmAnalysis = am.getAnalysis<LoopNestAnalysis>();
  op->walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
    propagateSCFFor(forOp);
    auto maybeLoopNest = lmAnalysis.getLoopNest(forOp);
    if (maybeLoopNest.hasValue()) {
      propagateLoopNest(maybeLoopNest.getValue());
    }
  });
  op->walk(
      [&](linalg::GenericOp genericOp) { propagateLinalgGeneric(genericOp); });
  op->walk([&](tensor::InsertSliceOp insertSliceOp) {
    propagateInsertSlice(insertSliceOp);
  });
}

// These getSparsityType functions the same name but very different functions
Optional<HotSparsityType>
SparsePropagation::getSparsityEncoding(RankedTensorType type) const {
  if (auto encoding = type.getEncoding().dyn_cast_or_null<StringAttr>()) {
    return StringSwitch<Optional<HotSparsityType>>(encoding.getValue())
        .Case("empty", HotSparsityType::Empty)
        .Case("onehot", HotSparsityType::OneHot)
        .Case("rowhot", HotSparsityType::RowHot)
        .Case("colhot", HotSparsityType::ColHot)
        .Default(llvm::None);
  }
  return llvm::None;
}

void SparsePropagation::setIndices(Value tensor, Value indices) {
  this->indices[tensor] = indices;
}

Optional<Value> SparsePropagation::getIndices(Value tensor) {
  return indices[tensor] ? Optional<Value>(indices[tensor]) : llvm::None;
}

Optional<HotSparsityType> SparsePropagation::getSparsityType(Value val) const {
  if (sparsityTypes.count(val) == 0) {
    return llvm::None;
  }
  return sparsityTypes.lookup(val);
}

void SparsePropagation::propagateInsertSlice(tensor::InsertSliceOp op) {
  Optional<HotSparsityType> sourceSparsity = getSparsityType(op.source());

  if (!(sourceSparsity.hasValue() &&
        sparsityTypes[op.dest()] == HotSparsityType::Empty)) {
    return;
  }

  sparsityTypes[op.result()] = sourceSparsity.getValue();
}

void SparsePropagation::propagateLinalgGeneric(linalg::GenericOp op) {
  // TODO: Reduce code duplication with sparse codegen
  auto isSparse = [this](OpOperand *operand) {
    return getSparsityType(operand->get()).hasValue();
  };
  if (llvm::count_if(op.getInputOperands(), isSparse) != 1) {
    return;
  }
  OpOperand *sparseOperand = *llvm::find_if(op.getInputOperands(), isSparse);
  HotSparsityType spType = getSparsityType(sparseOperand->get()).getValue();
  AffineMap sparseMap = op.getTiedIndexingMap(sparseOperand);
  DenseSet<unsigned> sparseDims;
  switch (spType) {
  case HotSparsityType::OneHot:
    for (auto result : sparseMap.getResults()) {
      sparseDims.insert(result.cast<AffineDimExpr>().getPosition());
    }
    break;
  case HotSparsityType::ColHot:
    sparseDims.insert(
        sparseMap.getResult(1).cast<AffineDimExpr>().getPosition());
    break;
  case HotSparsityType::RowHot:
    sparseDims.insert(
        sparseMap.getResult(0).cast<AffineDimExpr>().getPosition());
    break;
  default:
    break;
  }

  for (OpOperand *output : op.getOutputOperands()) {
    AffineMap map = op.getTiedIndexingMap(output);
    SmallVector<bool, 4> sparseMask;
    for (auto result : map.getResults()) {
      sparseMask.push_back(
          sparseDims.contains(result.cast<AffineDimExpr>().getPosition()));
    }
    if (llvm::all_of(sparseMask, [](bool pred) { return pred; })) {
      sparsityTypes[op.getTiedOpResult(output)] = HotSparsityType::OneHot;
    } else if (sparseMask.size() == 2 && sparseMask[0] && !sparseMask[1]) {
      sparsityTypes[op.getTiedOpResult(output)] = HotSparsityType::RowHot;
    } else if (sparseMask.size() == 2 && !sparseMask[0] && sparseMask[1]) {
      sparsityTypes[op.getTiedOpResult(output)] = HotSparsityType::ColHot;
    }
  }
}

void SparsePropagation::propagateSCFFor(scf::ForOp op) {
  for (OpOperand &operand : op.getIterOpOperands()) {
    auto spType = getSparsityType(operand.get());
    if (spType.hasValue()) {
      sparsityTypes[op.getRegionIterArgForOpOperand(operand)] =
          spType.getValue();
    }
  }
}

void SparsePropagation::propagateLoopNest(LoopNest loopNest) {
  if (loopNest.inputTensorOperands.size() == 1 &&
      loopNest.outputTensorOperands.size() == 1 &&
      loopNest.inputMaps[0].isIdentity()) {
    Value inputOperand = loopNest.inputTensorOperands.front();
    auto spType = getSparsityType(inputOperand);
    auto destSpType = getSparsityType(loopNest.outputTensorOperands.front());
    if (spType.hasValue() && spType.getValue() == HotSparsityType::OneHot &&
        destSpType.hasValue() && destSpType == HotSparsityType::Empty) {
      sparsityTypes[loopNest.results.front()] = spType.getValue();
    }
  }
}

} // namespace mlir

namespace {
using namespace mlir;

RankedTensorType stripEncoding(RankedTensorType sourceType) {
  return RankedTensorType::get(sourceType.getShape(),
                               sourceType.getElementType());
}

class SparsifyFuncOp : public OpConversionPattern<FuncOp> {
private:
  SparsePropagation &spAnalysis;
  bool hasRecognizedEncoding(Type type) const {
    if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
      return spAnalysis.getSparsityEncoding(rankedTensorType).hasValue();
    }
    return false;
  }

  FunctionType stripEncodingFromFunc(FunctionType funcTyp) const {
    SmallVector<Type> inputTypes;
    for (auto typ : funcTyp.getInputs()) {
      if (hasRecognizedEncoding(typ)) {
        inputTypes.push_back(stripEncoding(typ.cast<RankedTensorType>()));
      } else {
        inputTypes.push_back(typ);
      }
    }
    SmallVector<Type> outputTypes;
    for (auto typ : funcTyp.getResults()) {
      if (hasRecognizedEncoding(typ)) {
        outputTypes.push_back(stripEncoding(typ.cast<RankedTensorType>()));
      } else {
        outputTypes.push_back(typ);
      }
    }
    return FunctionType::get(funcTyp.getContext(), inputTypes,
                             funcTyp.getResults());
  }

public:
  SparsifyFuncOp(SparsePropagation &spAnalysis, TypeConverter &typeConverter,
                 MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1),
        spAnalysis{spAnalysis} {}

  LogicalResult
  matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (llvm::none_of(op.getArguments(), [this](BlockArgument arg) {
          return spAnalysis.getSparsityType(arg).hasValue() &&
                 !spAnalysis.getIndices(arg).hasValue();
        })) {
      return failure();
    }

    size_t idx = 0;
    auto *ctx = rewriter.getContext();

    rewriter.updateRootInPlace(op, [&]() {
      op.setType(stripEncodingFromFunc(op.getType()));
      for (BlockArgument arg : op.getArguments()) {
        auto spType = spAnalysis.getSparsityType(arg);
        if (spType.hasValue()) {
          auto addIndexArgument = [&]() {
            Type indicesType;
            switch (spType.getValue()) {
            case HotSparsityType::OneHot: {
              int64_t rank = arg.getType().cast<ShapedType>().getRank();
              if (rank == 1) {
                indicesType = IndexType::get(ctx);
              } else {
                indicesType = MemRefType::get({rank}, IndexType::get(ctx));
              }
              break;
            }
            case HotSparsityType::RowHot:
              LLVM_FALLTHROUGH;
            case HotSparsityType::ColHot:
              // Might want to switch this to use a memref for uniformity
              indicesType = IndexType::get(ctx);
              break;
            case HotSparsityType::Empty:
              return;
            }
            auto originalArgType = arg.getType().cast<RankedTensorType>();
            auto argType = RankedTensorType::get(
                originalArgType.getShape(), originalArgType.getElementType());
            arg.setType(argType);
            op.insertArgument(idx + 1, indicesType, {});
            spAnalysis.setIndices(arg, op.getArgument(idx + 1));
          };
          addIndexArgument();

          idx++;
        }
        idx++;
      }
    });

    return success();
  }
};

bool matchSparsifyForOp(scf::ForOp op, SparsePropagation &spAnalysis,
                        LoopNestAnalysis &lnAnalysis) {
  auto maybeLoopNest = lnAnalysis.getLoopNest(op);
  if (!maybeLoopNest.hasValue()) {
    return false;
  }
  LoopNest loopNest = maybeLoopNest.getValue();
  return llvm::any_of(loopNest.inputTensorOperands, [&spAnalysis](Value val) {
    auto spType = spAnalysis.getSparsityType(val);
    return spType.hasValue() && spType.getValue() != HotSparsityType::Empty;
  });
}

bool matchSparsifyGenericOp(linalg::GenericOp op,
                            SparsePropagation &spAnalysis) {
  bool hasEncoding =
      llvm::any_of(op.getInputOperands(), [&spAnalysis](OpOperand *operand) {
        auto spType = spAnalysis.getSparsityType(operand->get());
        return spType.hasValue() && spType.getValue() != HotSparsityType::Empty;
      });
  // TODO: potentially dangerous to not use this. We currently need it to match
  // a matmul in hand tracking because the zero value is propagated through loop
  // iter args.

  // bool outputIsZero =
  //     op.getNumOutputs() == 1 &&
  //     spAnalysis
  //             .getSparsityType(op.getOutputOperand(0)->get())
  //             // We need this to just be a sparsity type other than empty.
  //             .getValueOr(HotSparsityType::OneHot) == HotSparsityType::Empty;
  // if (op->hasAttr("debugme")) {
  //   errs() << "debugme found. Output is zero: " << outputIsZero << "\n";
  //   for (OpOperand *operand : op.getInputOperands()) {
  //     if (spAnalysis.getSparsityType(operand->get()).hasValue()) {
  //       errs() << "operand is sparse: "
  //              << spAnalysis.getSparsityType(operand->get()).getValue() <<
  //              "\n";
  //     }
  //   }
  // }
  return hasEncoding;
}

// Currently only works for 2D one-hot case.
SmallVector<Value> convertIndicesToValues(Location loc, Value memrefIndices,
                                          OpBuilder &builder) {
  if (memrefIndices.getType().isa<IndexType>()) {
    return {memrefIndices};
  }
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  return {builder.create<memref::LoadOp>(loc, memrefIndices, zero),
          builder.create<memref::LoadOp>(loc, memrefIndices, one)};
}

Value inlineRegion(linalg::GenericOp op, ValueRange indexedValues,
                   OpBuilder &builder) {
  auto &block = op->getRegion(0).front();
  BlockAndValueMapping map;
  map.map(block.getArguments(), indexedValues);
  for (auto &op : block.without_terminator()) {
    auto *newOp = builder.clone(op, map);
    map.map(op.getResults(), newOp->getResults());
  }

  Operation *terminator = block.getTerminator();
  Value toStore = map.lookupOrDefault(terminator->getOperand(0));
  return toStore;
}

static SmallVector<Value> makeCanonicalAffineApplies(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ArrayRef<Value> vals) {
  if (map.isEmpty())
    return {};

  assert(map.getNumInputs() == vals.size());
  SmallVector<Value> res;
  res.reserve(map.getNumResults());
  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    auto exprMap = AffineMap::get(dims, map.getNumSymbols(), e);
    SmallVector<Value> operands(vals.begin(), vals.end());
    canonicalizeMapAndOperands(&exprMap, &operands);
    res.push_back(b.create<AffineApplyOp>(loc, exprMap, operands));
  }
  return res;
}

static Value getSizeOfLoop(OpBuilder &b, linalg::LinalgOp op,
                           unsigned position) {
  for (OpOperand *operand : op.getInputAndOutputOperands()) {
    AffineMap map = op.getTiedIndexingMap(operand);
    if (map.isFunctionOfDim(position)) {
      for (auto pair : llvm::enumerate(map.getResults())) {
        if (pair.value().isFunctionOfDim(position) &&
            pair.value().isa<AffineDimExpr>()) {
          return b.create<tensor::DimOp>(op.getLoc(), operand->get(),
                                         pair.index());
        }
      }
    }
  }
  llvm_unreachable("Failed to find size of linalg op loop");
}

class SparsifyLinalgGenericOp : public OpConversionPattern<linalg::GenericOp> {
private:
  SparsePropagation &spAnalysis;

  Value allocateOutputSpace(Value tensor, Location loc, OpBuilder &b) const {
    RankedTensorType resultType = tensor.getType().cast<RankedTensorType>();
    BufferizeTypeConverter typeConverter;
    SmallVector<Value> dynamicSizes;
    dynamicSizes.reserve(resultType.getNumDynamicDims());
    for (unsigned idx = 0; idx < resultType.getRank(); idx++) {
      if (resultType.isDynamicDim(idx)) {
        dynamicSizes.push_back(b.create<tensor::DimOp>(loc, tensor, idx));
      }
    }
    return b.create<memref::AllocOp>(
        loc, typeConverter.convertType(resultType).cast<MemRefType>(),
        dynamicSizes);
  }

public:
  SparsifyLinalgGenericOp(SparsePropagation &spAnalysis, MLIRContext *ctx)
      : OpConversionPattern(ctx, /*benefit=*/1), spAnalysis{spAnalysis} {}
  LogicalResult
  matchAndRewrite(linalg::GenericOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matchSparsifyGenericOp(op, spAnalysis)) {
      return failure();
    }
    auto isSparse = [this](OpOperand *operand) {
      return spAnalysis.getSparsityType(operand->get()).hasValue();
    };
    OpOperand *sparseOperand = *llvm::find_if(op.getInputOperands(), isSparse);
    assert(llvm::count_if(op.getInputOperands(), isSparse) == 1 &&
           "Expected exactly 1 operand to have hot sparsity type (not yet "
           "implemented)");
    HotSparsityType spType =
        spAnalysis.getSparsityType(sparseOperand->get()).getValue();
    AffineMap sparseMap = op.getTiedIndexingMap(sparseOperand);
    DenseSet<unsigned> sparseDims;
    switch (spType) {
    case HotSparsityType::OneHot:
      for (auto result : sparseMap.getResults()) {
        sparseDims.insert(result.cast<AffineDimExpr>().getPosition());
      }
      break;
    case HotSparsityType::ColHot:
      sparseDims.insert(
          sparseMap.getResult(1).cast<AffineDimExpr>().getPosition());
      break;
    case HotSparsityType::RowHot:
      sparseDims.insert(
          sparseMap.getResult(0).cast<AffineDimExpr>().getPosition());
      break;
    default:
      break;
    }
    SmallVector<Value, 4> lbs, ubs, steps;
    Location loc = op.getLoc();
    BufferizeTypeConverter typeConverter;
    Value zero = rewriter.create<arith::ConstantOp>(
        loc,
        FloatAttr::get(op.getOutputTensorTypes()[0].getElementType(), 0.0));
    Value idxZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value idxOne = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    // Value space = rewriter.create<memref::BufferCastOp>(
    //     loc, typeConverter.convertType(op.getOutputTensorTypes()[0]),
    //     op.getOutputOperand(0)->get());
    Value space =
        allocateOutputSpace(op.getOutputOperand(0)->get(), loc, rewriter);
    rewriter.create<linalg::FillOp>(loc, zero, space);
    for (unsigned dim = 0; dim < sparseMap.getNumDims(); dim++) {
      if (!sparseDims.contains(dim)) {
        lbs.push_back(idxZero);
        ubs.push_back(getSizeOfLoop(rewriter, op, dim));
        steps.push_back(idxOne);
      }
    }

    Optional<Value> sparseIndices = spAnalysis.getIndices(sparseOperand->get());
    if (!sparseIndices.hasValue()) {
      op.emitError() << "sparse op was missing indices\n";
      return failure();
    }

    // The sparse induction vars must correspond to the dimensions of the sparse
    // operand, hence the null value in the col-hot case.
    SmallVector<Value> sparseIvs =
        spType == HotSparsityType::OneHot
            ? ValueRange{convertIndicesToValues(loc, sparseIndices.getValue(),
                                                rewriter)}
        : spType == HotSparsityType::ColHot
            ? ValueRange{Value(), sparseIndices.getValue()}
            : ValueRange{sparseIndices.getValue()};

    scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          unsigned denseIdx = 0;
          SmallVector<Value> allIvs;
          allIvs.reserve(sparseMap.getNumDims());
          for (unsigned dim = 0; dim < sparseMap.getNumDims(); dim++) {
            if (sparseDims.contains(dim)) {
              for (auto pair : llvm::enumerate(sparseMap.getResults())) {
                if (pair.value().isFunctionOfDim(dim)) {
                  allIvs.push_back(sparseIvs[pair.index()]);
                  break;
                }
              }
            } else {
              allIvs.push_back(ivs[denseIdx]);
              denseIdx++;
            }
          }

          SmallVector<Value, 3> indexedValues;
          indexedValues.reserve(op.getNumInputsAndOutputs());
          // Emit reads to input and output views
          for (OpOperand *operand : op.getInputAndOutputOperands()) {
            indexedValues.push_back(builder.create<tensor::ExtractOp>(
                loc, operand->get(),
                makeCanonicalAffineApplies(
                    builder, loc, op.getTiedIndexingMap(operand), allIvs)));
          }
          // Inline region and emit store
          Value toStore = inlineRegion(op, indexedValues, builder);
          builder.create<memref::StoreOp>(
              loc, toStore, space,
              makeCanonicalAffineApplies(
                  builder, loc, op.getTiedIndexingMap(op.getOutputOperand(0)),
                  allIvs));
        });
    rewriter.replaceOpWithNewOp<memref::TensorLoadOp>(op, space);

    // propagate sparse indices
    for (OpOperand *outOperand : op.getOutputOperands()) {
      SmallVector<Value, 2> newSparseIdxVals;
      AffineMap outMap = op.getTiedIndexingMap(outOperand);
      for (auto result : outMap.getResults()) {
        unsigned dim = result.cast<AffineDimExpr>().getPosition();
        if (sparseDims.contains(dim)) {
          for (auto pair : llvm::enumerate(sparseMap.getResults())) {
            if (pair.value().isFunctionOfDim(dim)) {
              newSparseIdxVals.push_back(sparseIvs[pair.index()]);
              break;
            }
          }
        }
      }

      if (newSparseIdxVals.size() == 1) {
        spAnalysis.setIndices(op.getTiedOpResult(outOperand),
                              newSparseIdxVals[0]);
      } else {
        auto newIndices = rewriter.create<memref::AllocaOp>(
            loc, MemRefType::get(newSparseIdxVals.size(),
                                 IndexType::get(rewriter.getContext())));
        for (auto pair : llvm::enumerate(newSparseIdxVals)) {
          Value spIdx =
              rewriter.create<arith::ConstantIndexOp>(loc, pair.index());
          rewriter.create<memref::StoreOp>(loc, pair.value(), newIndices,
                                           spIdx);
        }
        spAnalysis.setIndices(op.getTiedOpResult(outOperand), newIndices);
      }
    }
    return success();
  }
};

class SparsifyForOp : public OpConversionPattern<scf::ForOp> {
private:
  SparsePropagation &spAnalysis;
  LoopNestAnalysis &lnAnalysis;

  void stripEncodingFromLoop(scf::ForOp op) const {
    for (OpOperand &operand : op.getIterOpOperands()) {
      BlockArgument arg = op.getRegionIterArgForOpOperand(operand);
      OpResult result = op.getResultForOpOperand(operand);

      if (auto rankedTensorType = arg.getType().dyn_cast<RankedTensorType>()) {
        if (spAnalysis.getSparsityEncoding(rankedTensorType).hasValue()) {
          arg.setType(stripEncoding(rankedTensorType));
          result.setType(stripEncoding(rankedTensorType));
        }
      }
    }
    for (Operation &op : op.getLoopBody().getOps()) {
      for (OpResult result : op.getResults()) {
        if (auto rankedTensorType =
                result.getType().dyn_cast<RankedTensorType>()) {
          if (spAnalysis.getSparsityEncoding(rankedTensorType).hasValue()) {
            result.setType(stripEncoding(rankedTensorType));
          }
        }
      }
    }
  }

public:
  SparsifyForOp(SparsePropagation &spAnalysis, LoopNestAnalysis &lnAnalysis,
                MLIRContext *ctx)
      : OpConversionPattern(ctx, /*benefit=*/1), spAnalysis{spAnalysis},
        lnAnalysis{lnAnalysis} {}

  LogicalResult
  matchAndRewrite(scf::ForOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matchSparsifyForOp(op, spAnalysis, lnAnalysis)) {
      return failure();
    }
    LoopNest loopNest = lnAnalysis.getLoopNest(op).getValue();

    errs() << "sparsifying loop nest with "
           << loopNest.inputTensorOperands.size() << " inputs\n";
    for (auto pair : llvm::enumerate(loopNest.inputTensorOperands)) {
      Value inputOperand = pair.value();
      errs() << "input: " << inputOperand << "\n";
      if (auto spType = spAnalysis.getSparsityType(inputOperand)) {
        errs() << "sparsity type: " << spType.getValue() << "\n";
        errs() << "inputMap: " << loopNest.inputMaps[pair.index()] << "\n";
      } else {
        errs() << "debug\n";
        errs() << *inputOperand.getParentBlock()->getParentOp() << "\n";
      }
    }
    if (loopNest.inputTensorOperands.size() == 1 &&
        loopNest.inputMaps.front().isIdentity()) {
      auto spType =
          spAnalysis.getSparsityType(loopNest.inputTensorOperands.front());
      if (spType.hasValue() && spType.getValue() == HotSparsityType::OneHot) {
        Location loc = op.getLoc();
        BlockAndValueMapping map;
        Value sparseIndices =
            spAnalysis.getIndices(loopNest.inputTensorOperands.front())
                .getValue();
        map.map(loopNest.inductionVars,
                convertIndicesToValues(loc, sparseIndices, rewriter));
        map.map(loopNest.inputRegionArgs, loopNest.inputTensorOperands);
        map.map(loopNest.outputRegionArgs, loopNest.outputTensorOperands);

        for (auto &op : loopNest.loops.back().getBody()->without_terminator()) {
          if (loopNest.ivComputation.contains(&op)) {
            continue;
          }
          auto *newOp = rewriter.clone(op, map);
          for (OpResult result : newOp->getResults()) {
            if (auto rankedTensorType =
                    result.getType().dyn_cast<RankedTensorType>()) {
              if (spAnalysis.getSparsityEncoding(rankedTensorType).hasValue()) {
                result.setType(stripEncoding(rankedTensorType));
              }
            }
          }
          map.map(op.getResults(), newOp->getResults());
        }
        Operation *terminator =
            loopNest.loops.back().getBody()->getTerminator();
        SmallVector<Value> results;
        for (auto termOperand : terminator->getOperands()) {
          results.push_back(map.lookupOrDefault(termOperand));
        }

        for (auto tup :
             llvm::zip(llvm::make_range(results.begin() +
                                            loopNest.inputTensorOperands.size(),
                                        results.end()),
                       llvm::make_range(op.getResults().begin() +
                                            loopNest.inputTensorOperands.size(),
                                        op.getResults().end()))) {
          // Need to update indices here
          auto insertOp =
              cast<tensor::InsertOp>(std::get<0>(tup).getDefiningOp());
          OpResult originalResult = std::get<1>(tup);
          auto newIndices = rewriter.create<memref::AllocaOp>(
              op.getLoc(), sparseIndices.getType().cast<MemRefType>());
          for (auto pair : llvm::enumerate(insertOp.indices())) {
            Value spIdx =
                rewriter.create<arith::ConstantIndexOp>(loc, pair.index());
            rewriter.create<memref::StoreOp>(loc, pair.value(), newIndices,
                                             spIdx);
          }
          spAnalysis.setIndices(originalResult, newIndices);
        }
        rewriter.replaceOp(op, results);
        return success();
      }
    }

    rewriter.updateRootInPlace(op, [&]() {
      stripEncodingFromLoop(op);
      op.walk([&](scf::ForOp childLoop) { stripEncodingFromLoop(childLoop); });
    });

    return success();
  }
};

bool matchSparsifyInsertSlice(tensor::InsertSliceOp op,
                              SparsePropagation &spAnalysis) {
  Optional<HotSparsityType> spType = spAnalysis.getSparsityType(op.result());
  Optional<HotSparsityType> destSpType = spAnalysis.getSparsityType(op.dest());
  Optional<Value> indices = spAnalysis.getIndices(op.source());
  return spType.hasValue() && destSpType.hasValue() && indices.hasValue() &&
         destSpType.getValue() == HotSparsityType::Empty;
}

class SparsifyInsertSlice : public OpConversionPattern<tensor::InsertSliceOp> {
private:
  SparsePropagation &spAnalysis;
  bool isRowInsertion(tensor::InsertSliceOp op) const {
    auto inferredType =
        tensor::ExtractSliceOp::inferResultType(
            op.getType(), extractFromI64ArrayAttr(op.static_offsets()),
            extractFromI64ArrayAttr(op.static_sizes()),
            extractFromI64ArrayAttr(op.static_strides()))
            .cast<RankedTensorType>();
    Optional<llvm::SmallDenseSet<unsigned>> mask = computeRankReductionMask(
        inferredType.getShape(), op.getSourceType().getShape());
    if (!mask.hasValue()) {
      return false;
    }
    return !mask.getValue().contains(inferredType.getRank() - 1);
  }

public:
  SparsifyInsertSlice(SparsePropagation &spAnalysis, MLIRContext *ctx)
      : OpConversionPattern(ctx, /*benefit=*/1), spAnalysis{spAnalysis} {}

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matchSparsifyInsertSlice(op, spAnalysis)) {
      return failure();
    }
    Optional<HotSparsityType> spType = spAnalysis.getSparsityType(op.result());
    Optional<Value> indices = spAnalysis.getIndices(op.source());
    if (!indices.hasValue()) {
      op.emitError() << "sparse op was missing indices\n";
      return failure();
    }

    Location loc = op.getLoc();
    auto intToAttr = [&](int64_t i) {
      return IntegerAttr::get(IntegerType::get(rewriter.getContext(), 64), i);
    };
    switch (spType.getValue()) {
    case HotSparsityType::OneHot: {
      if (indices.getValue().getType().isa<IndexType>()) {
        assert(isRowInsertion(op) && "only row 1-d insertions supported");
        SmallVector<Value> valueOffsets;

        for (auto offset :
             ArrayRef<OpFoldResult>{op.getMixedOffsets()}.drop_back(1)) {
          if (offset.is<Value>()) {
            valueOffsets.push_back(offset.get<Value>());
          } else {
            valueOffsets.push_back(rewriter.create<arith::ConstantIndexOp>(
                loc, offset.get<Attribute>()
                         .cast<IntegerAttr>()
                         .getValue()
                         .getSExtValue()));
          }
        }

        APInt lastOffset = op.static_offsets()[op.static_offsets().size() - 1]
                               .cast<IntegerAttr>()
                               .getValue();
        if (lastOffset.isZero()) {
          valueOffsets.push_back(indices.getValue());
        } else {
          valueOffsets.push_back(rewriter.create<arith::AddIOp>(
              loc,
              rewriter.create<arith::ConstantIndexOp>(
                  loc, lastOffset.getSExtValue()),
              indices.getValue()));
          // TODO: propagate indices properly
          // spAnalysis
        }
        Value scalar = rewriter.create<tensor::ExtractOp>(loc, op.source(),
                                                          indices.getValue());
        rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, scalar, op.dest(),
                                                      valueOffsets);
      }
      break;
    }
    case HotSparsityType::RowHot: {
      SmallVector<OpFoldResult> offsets;
      SmallVector<OpFoldResult> sizes{op.getMixedSizes()};
      offsets.append({indices.getValue(), intToAttr(0)});
      sizes[0] = intToAttr(1);

      auto slice = rewriter.create<tensor::ExtractSliceOp>(
          loc, op.source(), offsets, sizes, op.getMixedStrides());
      assert(op.getMixedOffsets()[0]
                 .get<Attribute>()
                 .cast<IntegerAttr>()
                 .getValue()
                 .isZero() &&
             "nonzero row offset for row-hot insert not yet supported");
      rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
          op, slice, op.dest(), offsets, sizes, op.getMixedStrides());
      break;
    }
    default:
      errs() << op.getLoc() << "\n";
      errs() << "sparsify tensor.insert_slice for " << spType.getValue()
             << " not yet implemented\n";
      llvm_unreachable("Not yet implemented");
    }

    // propagate indices
    spAnalysis.setIndices(op.result(), indices.getValue());
    return success();
  }
};

struct StructuredSparsifyPass
    : public PassWrapper<StructuredSparsifyPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
  }
  StringRef getArgument() const override { return "structured-sparsify"; }
  StringRef getDescription() const override {
    return "Sparsify generated code with structured sparsity patterns (one "
           "hot, row hot, col hot)";
  }
  void runOnOperation() final {
    auto *context = &getContext();
    auto &loopNestAnalysis = getAnalysis<LoopNestAnalysis>();
    auto &sparsePropagation = getAnalysis<SparsePropagation>();
    RewritePatternSet patterns(context);

    // BufferizeTypeConverter typeConverter;
    TypeConverter typeConverter;
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
      return !matchSparsifyForOp(op, sparsePropagation, loopNestAnalysis);
    });
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
      return llvm::none_of(funcOp.getArguments(), [&sparsePropagation](
                                                      BlockArgument arg) {
        bool hasIndex = sparsePropagation.getIndices(arg).hasValue();
        return sparsePropagation.getSparsityType(arg).hasValue() && !hasIndex;
      });
    });
    target.addLegalOp<tensor::ExtractOp, tensor::InsertOp,
                      tensor::ExtractSliceOp, tensor::DimOp>();
    target.addDynamicallyLegalOp<tensor::InsertSliceOp>(
        [&](tensor::InsertSliceOp op) {
          return !matchSparsifyInsertSlice(op, sparsePropagation);
        });
    target.addLegalOp<linalg::FillOp, linalg::YieldOp>();
    target.addDynamicallyLegalOp<linalg::GenericOp>([&](linalg::GenericOp op) {
      return !matchSparsifyGenericOp(op, sparsePropagation);
    });

    patterns.add<SparsifyFuncOp>(sparsePropagation, typeConverter,
                                 patterns.getContext());
    patterns.add<SparsifyInsertSlice>(sparsePropagation, patterns.getContext());
    patterns.add<SparsifyLinalgGenericOp>(sparsePropagation,
                                          patterns.getContext());
    patterns.add<SparsifyForOp>(sparsePropagation, loopNestAnalysis,
                                patterns.getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::lagrad::createSparsifyPass() {
  return std::make_unique<StructuredSparsifyPass>();
}
