#include "Standalone/Analysis.h"
#include "Standalone/Passes.h"
#include "Standalone/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
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

SparsePropagation::SparsePropagation(Operation *op) {
  op->walk([&](FuncOp funcOp) { DEBUGpopulateFunc(debug_names, funcOp); });
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
  op->walk<WalkOrder::PreOrder>(
      [&](scf::ForOp forOp) { propagateSCFFor(forOp); });
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
  // TODO: This isn't strictly correct because the insert_slice could appear
  // with an offset.
  indices[op.result()] = indices[op.source()];
}

bool isElementwiseOp(linalg::GenericOp op) {
  SmallVector<AffineMap> indexingMaps = op.getIndexingMaps();
  auto identityMap = AffineMap::getMultiDimIdentityMap(
      indexingMaps.front().getNumDims(), op.getContext());
  bool hasOnlyIdentityIndexing =
      llvm::all_of(indexingMaps, [&identityMap](AffineMap map) {
        return map == identityMap;
      });
  bool hasOnlyParallelIterators =
      llvm::all_of(op.iterator_types(), [](Attribute attr) {
        return attr.cast<StringAttr>().getValue() ==
               getParallelIteratorTypeName();
      });
  return hasOnlyIdentityIndexing && hasOnlyParallelIterators;
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
    sparseDims.insert(
        sparseMap.getResult(0).cast<AffineDimExpr>().getPosition());
    sparseDims.insert(
        sparseMap.getResult(1).cast<AffineDimExpr>().getPosition());
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
            case HotSparsityType::OneHot:
              indicesType = MemRefType::get({2}, IndexType::get(ctx));
              break;
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

struct LoopNest {
  SmallVector<scf::ForOp> loops;
  SmallVector<Value> inductionVars, inputTensorOperands, inputRegionArgs,
      outputTensorOperands, outputRegionArgs;
  SmallVector<AffineMap> inputMaps;
  DenseSet<Operation *> ivComputation;
};

// Find the induction variable of a loop that may be iterating in reverse.
Value getInductionVar(scf::ForOp op, LoopNest &loopNest) {
  if (op.getInductionVar().hasOneUse()) {
    if (auto subiOp =
            dyn_cast<arith::SubIOp>(*op.getInductionVar().getUsers().begin())) {
      // case 1: (ub - iv) - 1
      if (subiOp.lhs() == op.upperBound() &&
          subiOp.rhs() == op.getInductionVar() &&
          subiOp.getResult().hasOneUse()) {
        if (auto secondSubiOp = dyn_cast<arith::SubIOp>(
                *subiOp.getResult().getUsers().begin())) {
          auto rhs = dyn_cast_or_null<arith::ConstantIndexOp>(
              secondSubiOp.rhs().getDefiningOp());
          if (secondSubiOp.lhs() == subiOp.getResult() && rhs &&
              rhs.value() == 1) {
            loopNest.ivComputation.insert(subiOp);
            loopNest.ivComputation.insert(secondSubiOp);
            return secondSubiOp.getResult();
          }
        }
      } else {
        // case 2: (const ub - 1) - iv
        auto constLHS = dyn_cast_or_null<arith::ConstantIndexOp>(
            subiOp.lhs().getDefiningOp());
        auto constUpperBound = dyn_cast_or_null<arith::ConstantIndexOp>(
            op.upperBound().getDefiningOp());
        if (subiOp.rhs() == op.getInductionVar() && constLHS &&
            constUpperBound &&
            constLHS.value() == constUpperBound.value() - 1) {
          loopNest.ivComputation.insert(subiOp);
          return subiOp.getResult();
        }
      }
    }
  }
  return op.getInductionVar();
}

bool matchSparsifyForOp(scf::ForOp op, SparsePropagation &spAnalysis,
                        LoopNest &loopNest) {
  if (op->getParentOfType<scf::ForOp>()) {
    return false;
  }

  WalkResult result = op->walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
    auto childLoops = forOp.getBody()->getOps<scf::ForOp>();
    loopNest.loops.push_back(forOp);
    loopNest.inductionVars.push_back(getInductionVar(forOp, loopNest));
    if (childLoops.empty()) {
      // this is the innermost loop
      // need to maybe collect reads and writes
      if (!llvm::all_of(
              forOp.getBody()->getOperations(), [](Operation &childOp) {
                return isa<tensor::ExtractOp>(childOp) ||
                       isa<tensor::InsertOp>(childOp) ||
                       llvm::all_of(childOp.getResultTypes(), [](Type type) {
                         return type.isIntOrIndexOrFloat();
                       });
              })) {
        return WalkResult::interrupt();
      }
      if (llvm::none_of(
              forOp.getBody()->getOps<tensor::ExtractOp>(),
              [&](tensor::ExtractOp extractOp) {
                return spAnalysis
                    .getSparsityEncoding(
                        extractOp.tensor().getType().cast<RankedTensorType>())
                    .hasValue();
              })) {
        return WalkResult::interrupt();
      }
      // All yielded operands should be the result of insert ops
      for (Value yieldOperand :
           forOp.getBody()->getTerminator()->getOperands()) {
        auto insertOp =
            dyn_cast_or_null<tensor::InsertOp>(yieldOperand.getDefiningOp());
        if (!insertOp) {
          return WalkResult::interrupt();
        }
        // need to find all the reads that affect the scalar within the loop
        // body
        DenseSet<Operation *> activeReads;
        SmallVector<Value> frontier{insertOp.scalar()};
        while (!frontier.empty()) {
          Value val = frontier.pop_back_val();
          if (Operation *definingOp = val.getDefiningOp()) {
            if (auto extractOp = dyn_cast<tensor::ExtractOp>(definingOp)) {
              activeReads.insert(extractOp);
            } else {
              for (auto operand : definingOp->getOperands()) {
                frontier.push_back(operand);
              }
            }
          }
        }

        for (auto read : activeReads) {
          auto extractOp = cast<tensor::ExtractOp>(read);
          if (extractOp.tensor() == insertOp.dest()) {
            // This extract op is an output operand
            loopNest.outputRegionArgs.push_back(extractOp.tensor());
            // TODO: reduce code duplication, this is a simpler bottom-up DFS
            Value tensor = extractOp.tensor();
            while (auto parentForOp = dyn_cast<scf::ForOp>(
                       tensor.getParentRegion()->getParentOp())) {
              auto iterOperand = llvm::find_if(
                  parentForOp.getIterOpOperands(), [&](OpOperand &operand) {
                    return parentForOp.getRegionIterArgForOpOperand(operand) ==
                           tensor;
                  });
              tensor = iterOperand->get();
            }
            loopNest.outputTensorOperands.push_back(tensor);
            continue;
          } else {
            SmallVector<AffineExpr, 4> resultExprs;
            for (auto idxVal : extractOp.indices()) {
              ptrdiff_t idx = std::distance(
                  loopNest.inductionVars.begin(),
                  std::find(loopNest.inductionVars.begin(),
                            loopNest.inductionVars.end(), idxVal));
              if (idx == static_cast<long>(loopNest.inductionVars.size())) {
                // The read indices were not a function of the induction vars
                return WalkResult::interrupt();
              }
              resultExprs.push_back(getAffineDimExpr(idx, op.getContext()));
            }
            loopNest.inputRegionArgs.push_back(extractOp.tensor());
            loopNest.inputMaps.push_back(AffineMap::get(
                loopNest.inductionVars.size(),
                /*symbolCount=*/0, resultExprs, op.getContext()));
            // extractOp.tensor().getParentRegion()->getParentOp()
            Value tensor = extractOp.tensor();
            while (auto parentForOp = dyn_cast<scf::ForOp>(
                       tensor.getParentRegion()->getParentOp())) {
              auto iterOperand = llvm::find_if(
                  parentForOp.getIterOpOperands(), [&](OpOperand &operand) {
                    return parentForOp.getRegionIterArgForOpOperand(operand) ==
                           tensor;
                  });
              tensor = iterOperand->get();
            }
            loopNest.inputTensorOperands.push_back(tensor);
          }
        }
      }
    } else {
      int numLoops = std::distance(childLoops.begin(), childLoops.end());
      if (numLoops > 1 ||
          !llvm::all_of(
              forOp.getBody()->getOperations(), [](Operation &childOp) {
                return isa<scf::ForOp>(childOp) ||
                       llvm::all_of(childOp.getResultTypes(), [](Type type) {
                         return type.isIntOrIndex();
                       });
              })) {
        return WalkResult::interrupt();
      }
      scf::ForOp childLoop = *childLoops.begin();
      if (!llvm::all_of_zip(forOp.getBody()->getTerminator()->getOperands(),
                            childLoop.getResults(),
                            [](Value yieldOperand, Value childResult) {
                              return yieldOperand == childResult;
                            })) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return false;
  }
  // errs() << "matched loop nest: " << op << "\n";
  return true;
}

bool matchSparsifyGenericOp(linalg::GenericOp op,
                            SparsePropagation &spAnalysis) {
  bool hasEncoding =
      llvm::any_of(op.getInputOperands(), [&spAnalysis](OpOperand *operand) {
        return spAnalysis.getSparsityType(operand->get()).hasValue();
      });
  bool outputIsZero =
      op.getNumOutputs() == 1 &&
      spAnalysis
              .getSparsityType(op.getOutputOperand(0)->get())
              // We need this to just be a sparsity type other than empty.
              .getValueOr(HotSparsityType::OneHot) == HotSparsityType::Empty;
  return hasEncoding && outputIsZero;
}

// Currently only works for one-hot case.
SmallVector<Value> convertIndicesToValues(Location loc, Value memrefIndices,
                                          OpBuilder &builder) {
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
      sparseDims.insert(
          sparseMap.getResult(0).cast<AffineDimExpr>().getPosition());
      sparseDims.insert(
          sparseMap.getResult(1).cast<AffineDimExpr>().getPosition());
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
    Value space = rewriter.create<memref::AllocOp>(
        loc,
        typeConverter.convertType(op.getResultTypes()[0]).cast<MemRefType>());
    rewriter.create<linalg::FillOp>(loc, zero, space);
    for (unsigned dim = 0; dim < sparseMap.getNumDims(); dim++) {
      if (!sparseDims.contains(dim)) {
        lbs.push_back(idxZero);
        ubs.push_back(getSizeOfLoop(rewriter, op, dim));
        steps.push_back(idxOne);
      }
    }

    ValueRange sparseIvs =
        spType == HotSparsityType::OneHot
            ? convertIndicesToValues(
                  loc, spAnalysis.getIndices(sparseOperand->get()).getValue(),
                  rewriter)
            : llvm::makeArrayRef(
                  spAnalysis.getIndices(sparseOperand->get()).getValue());
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
    return success();
  }
};

class SparsifyForOp : public OpConversionPattern<scf::ForOp> {
private:
  SparsePropagation &spAnalysis;

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
  SparsifyForOp(SparsePropagation &spAnalysis, MLIRContext *ctx)
      : OpConversionPattern(ctx, /*benefit=*/1), spAnalysis{spAnalysis} {}
  LogicalResult
  matchAndRewrite(scf::ForOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    LoopNest loopNest;
    if (!matchSparsifyForOp(op, spAnalysis, loopNest)) {
      return failure();
    }

    if (loopNest.inputTensorOperands.size() == 1 &&
        loopNest.inputMaps.front().isIdentity()) {
      auto spType =
          spAnalysis.getSparsityType(loopNest.inputTensorOperands.front());
      if (spType.hasValue() && spType.getValue() == HotSparsityType::OneHot) {
        Location loc = op.getLoc();
        BlockAndValueMapping map;
        map.map(loopNest.inductionVars,
                convertIndicesToValues(
                    loc,
                    spAnalysis.getIndices(loopNest.inputTensorOperands.front())
                        .getValue(),
                    rewriter));
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

    Location loc = op.getLoc();
    auto intToAttr = [&](int64_t i) {
      return IntegerAttr::get(IntegerType::get(rewriter.getContext(), 64), i);
    };
    switch (spType.getValue()) {
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
      llvm_unreachable("Not yet implemented");
    }
    return success();
  }
};

struct StructuredSparsifyPass
    : public PassWrapper<StructuredSparsifyPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<sparse_tensor::SparseTensorDialect>();
    registry.insert<scf::SCFDialect>();
  }
  StringRef getArgument() const override { return "structured-sparsify"; }
  StringRef getDescription() const override {
    return "Sparsify generated code with structured sparsity patterns (one "
           "hot, row hot, col hot)";
  }
  void runOnOperation() final {
    auto *context = &getContext();
    auto &sparsePropagation = getAnalysis<SparsePropagation>();
    RewritePatternSet patterns(context);

    // BufferizeTypeConverter typeConverter;
    TypeConverter typeConverter;
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addDynamicallyLegalOp<scf::ForOp>(
        [&sparsePropagation](scf::ForOp op) {
          LoopNest unused;
          return !matchSparsifyForOp(op, sparsePropagation, unused);
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
    patterns.add<SparsifyForOp>(sparsePropagation, patterns.getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::Standalone::createSparsifyPass() {
  return std::make_unique<StructuredSparsifyPass>();
}
