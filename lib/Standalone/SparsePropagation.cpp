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

  op->walk(
      [&](linalg::GenericOp genericOp) { propagateLinalgGeneric(genericOp); });
  op->walk([&](tensor::InsertSliceOp insertSliceOp) {
    propagateInsertSlice(insertSliceOp);
  });
}

// These getSparsityType functions the same name but very different functions
Optional<HotSparsityType>
SparsePropagation::getSparsityEncoding(RankedTensorType type) {
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

Optional<HotSparsityType> SparsePropagation::getSparsityType(Value val) {
  if (sparsityTypes.count(val) == 0) {
    return llvm::None;
  }
  return sparsityTypes[val];
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
  // For now, just code some special cases that we know we encounter. We can
  // make it more general later.
  if (op.getNumInputs() == 1 &&
      getSparsityType(op.getInputOperand(0)->get()).hasValue() &&
      isElementwiseOp(op)) {
    sparsityTypes[op.getResult(0)] =
        getSparsityType(op.getInputOperand(0)->get()).getValue();
    indices[op.getResult(0)] = indices[op.getOperand(0)];
  }
}
} // namespace mlir

namespace {
using namespace mlir;

class SparsifyFuncOp : public OpConversionPattern<FuncOp> {
private:
  SparsePropagation &spAnalysis;
  bool hasRecognizedEncoding(Type type) const {
    if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
      return spAnalysis.getSparsityEncoding(rankedTensorType).hasValue();
    }
    return false;
  }

  RankedTensorType stripEncoding(RankedTensorType sourceType) const {
    return RankedTensorType::get(sourceType.getShape(),
                                 sourceType.getElementType());
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
  return hasEncoding && outputIsZero && isElementwiseOp(op);
}

SmallVector<Value> convertIndicesToValues(Location loc, Value memrefIndices,
                                          OpBuilder &builder) {
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  return {builder.create<memref::LoadOp>(loc, memrefIndices, zero),
          builder.create<memref::LoadOp>(loc, memrefIndices, one)};
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
    if (isElementwiseOp(op)) {
      HotSparsityType spType =
          spAnalysis.getSparsityType(op.getResult(0)).getValue();
      switch (spType) {
      case HotSparsityType::OneHot: {
        // Convert indices to SSA
        Value operand = op.getOperand(0);
        Value output = op.getOutputOperand(0)->get();
        // Emit reads (to both input and output)
        Location loc = op.getLoc();
        ValueRange indexing = convertIndicesToValues(
            loc, spAnalysis.getIndices(operand).getValue(), rewriter);
        ValueRange indexedValues{
            rewriter.create<tensor::ExtractOp>(loc, operand, indexing),
            rewriter.create<tensor::ExtractOp>(loc, output, indexing)};

        // inline region and emit store
        auto &block = op->getRegion(0).front();
        BlockAndValueMapping map;
        map.map(block.getArguments(), indexedValues);
        for (auto &op : block.without_terminator()) {
          auto *newOp = rewriter.clone(op, map);
          map.map(op.getResults(), newOp->getResults());
        }

        Operation *terminator = block.getTerminator();
        Value toStore = map.lookupOrDefault(terminator->getOperand(0));
        Value result =
            rewriter.create<tensor::InsertOp>(loc, toStore, output, indexing);

        rewriter.replaceOp(op, result);
        return success();
      }
      default:
        llvm_unreachable("not yet implemented");
      }
    }
    return failure();
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
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
      return llvm::none_of(funcOp.getArguments(), [&sparsePropagation](
                                                      BlockArgument arg) {
        bool hasIndex = sparsePropagation.getIndices(arg).hasValue();
        return sparsePropagation.getSparsityType(arg).hasValue() && !hasIndex;
      });
    });
    target.addLegalOp<tensor::ExtractOp, tensor::InsertOp,
                      tensor::ExtractSliceOp>();
    target.addDynamicallyLegalOp<tensor::InsertSliceOp>(
        [&](tensor::InsertSliceOp op) {
          return !matchSparsifyInsertSlice(op, sparsePropagation);
        });
    target.addDynamicallyLegalOp<linalg::GenericOp>([&](linalg::GenericOp op) {
      return !matchSparsifyGenericOp(op, sparsePropagation);
    });

    patterns.add<SparsifyFuncOp>(sparsePropagation, typeConverter,
                                 patterns.getContext());
    patterns.add<SparsifyInsertSlice>(sparsePropagation, patterns.getContext());
    patterns.add<SparsifyLinalgGenericOp>(sparsePropagation,
                                          patterns.getContext());
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
