/**
 * A pass to lower linalg.generic ops that operate on triangular operands to
 * loops.
 */
#include "LAGrad/Passes.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
bool hasRecognizedEncoding(linalg::GenericOp op) {
  for (auto operand : op.getOperands()) {
    auto opType = operand.getType().dyn_cast_or_null<RankedTensorType>();
    if (opType) {
      auto encoding = opType.getEncoding();
      auto val = encoding.dyn_cast_or_null<StringAttr>();
      if (val && val.getValue() == "ltri") {
        return true;
      }
    }
  }
  return false;
}

FunctionType stripEncodingFromFunc(FunctionType funcTyp) {
  SmallVector<Type> inputTypes;
  for (auto typ : funcTyp.getInputs()) {
    auto rtt = typ.dyn_cast_or_null<RankedTensorType>();
    if (rtt) {
      inputTypes.push_back(
          RankedTensorType::get(rtt.getShape(), rtt.getElementType()));
    } else {
      inputTypes.push_back(typ);
    }
  }
  SmallVector<Type> outputTypes;
  for (auto typ : funcTyp.getResults()) {
    auto rtt = typ.dyn_cast_or_null<RankedTensorType>();
    if (rtt) {
      outputTypes.push_back(
          RankedTensorType::get(rtt.getShape(), rtt.getElementType()));
    } else {
      outputTypes.push_back(typ);
    }
  }
  return FunctionType::get(funcTyp.getContext(), inputTypes, outputTypes);
}

void eraseTriangularEncoding(Value operand, PatternRewriter &rewriter) {
  PatternRewriter::InsertionGuard insertionGuard(rewriter);
  auto tensorType = operand.getType().dyn_cast_or_null<RankedTensorType>();
  if (tensorType) {
    auto resultTensorType = RankedTensorType::get(tensorType.getShape(),
                                                  tensorType.getElementType());
    auto encoding = tensorType.getEncoding().dyn_cast_or_null<StringAttr>();
    if (encoding && encoding.getValue() == "ltri") {
      operand.setType(resultTensorType);
      auto definingOp = operand.getDefiningOp();
      if (definingOp && dyn_cast_or_null<arith::ConstantOp>(definingOp)) {
        auto constOp = dyn_cast<arith::ConstantOp>(definingOp);
        auto attr = constOp.getValueAttr();
        if (attr.isa<DenseElementsAttr>()) {
          auto dattr = attr.cast<DenseElementsAttr>();
          assert(dattr.isSplat() && "triangular loops for non-splatted dense "
                                    "tensors not yet supported");
          if (dattr.isSplat()) {
            llvm_unreachable("mid-refactoring, this is an unsupported case");
            // rewriter.setInsertionPoint(constOp);
            // rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            //     constOp, DenseElementsAttr::get(resultTensorType,
            //                                     dattr.getSplatValue()));
          }
        }
        return;
      }

      // for (auto &use : operand.getUses()) {
      //   llvm::outs() << "operand use: " << use.get() << "\n";
      // }

      auto parent = operand.getParentRegion()->getParentOp();
      if (parent && dyn_cast_or_null<func::FuncOp>(parent)) {
        auto funcOp = dyn_cast<func::FuncOp>(parent);
        SmallVector<Type> argumentTypes;
        int arg_index = -1;
        int index = 0;
        for (auto arg : funcOp.getArguments()) {
          if (arg == operand) {
            argumentTypes.push_back(operand.getType());
            arg_index = index;
          } else {
            argumentTypes.push_back(arg.getType());
          }
          index++;
        }
        // funcOp.setType(FunctionType::get(func::FuncOp.getContext(),
        // argumentTypes,
        //                                  funcOp.getType().getResults()));
        funcOp.setType(stripEncodingFromFunc(funcOp.getFunctionType()));
        auto uses = funcOp.getSymbolUses(funcOp->getParentOfType<ModuleOp>());
        if (uses.has_value()) {
          for (auto use : uses.value()) {
            for (auto useOperand : use.getUser()->getOperands()) {
              eraseTriangularEncoding(useOperand, rewriter);
            }
            for (auto result : use.getUser()->getResults()) {
              eraseTriangularEncoding(result, rewriter);
            }
          }
        }
      }
    }
  }
}

/// Taken from mlir/Dialect/Linalg/Utils/Utils.cpp
/// Given a list of subview ranges, extract individual values for lower, upper
/// bounds and steps and put them into the corresponding vectors.
static void unpackRanges(OpBuilder &builder, Location loc,
                         ArrayRef<Range> ranges, SmallVectorImpl<Value> &lbs,
                         SmallVectorImpl<Value> &ubs,
                         SmallVectorImpl<Value> &steps) {
  for (Range range : ranges) {
    lbs.emplace_back(
        getValueOrCreateConstantIndexOp(builder, loc, range.offset));
    ubs.emplace_back(getValueOrCreateConstantIndexOp(builder, loc, range.size));
    steps.emplace_back(
        getValueOrCreateConstantIndexOp(builder, loc, range.stride));
  }
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
                                                   linalg::LinalgOp linalgOp) {
  assert(iterArgs.size() == static_cast<size_t>(linalgOp.getNumDpsInits()) &&
         "Expected # of iter args to be equal to # of output tensor operands.");

  SmallVector<Value> indexedValues;
  indexedValues.reserve(linalgOp->getNumOperands());

  auto allIvsPlusDims = SmallVector<Value>(allIvs.begin(), allIvs.end());
  // 1.a. Emit load from input operands or for scalars access the operand
  // itself.
  for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
    if (linalgOp.isScalar(inputOperand)) {
      indexedValues.push_back(inputOperand->get());
      continue;
    }
    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(inputOperand), allIvsPlusDims);
    indexedValues.push_back(
        b.create<tensor::ExtractOp>(loc, inputOperand->get(), indexing));
  }

  // 1.b. Emit load from output views.
  for (auto pair : llvm::zip(linalgOp.getDpsInitOperands(), iterArgs)) {
    auto outputOperand = std::get<0>(pair);
    Value iterArg = std::get<1>(pair);

    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(outputOperand), allIvsPlusDims);
    indexedValues.push_back(
        b.create<tensor::ExtractOp>(loc, iterArg, indexing));
  }

  // 2. Inline region, currently only works for a single basic block.
  // 3. Emit store.
  SmallVector<Value> outputTensors{iterArgs};
  SmallVector<SmallVector<Value>, 8> indexing;
  for (auto outputOperand : linalgOp.getDpsInitOperands()) {
    indexing.push_back(makeCanonicalAffineApplies(
        b, loc, linalgOp.getMatchingIndexingMap(outputOperand),
        allIvsPlusDims));
  }
  return inlineRegionAndEmitStore(b, loc, linalgOp, indexedValues, indexing,
                                  outputTensors);
}

class ConvertGenericOp : public RewritePattern {
public:
  ConvertGenericOp(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op);
    if (!linalgOp || !linalgOp.hasTensorSemantics()) {
      return failure();
    }
    auto genericOp =
        dyn_cast_or_null<linalg::GenericOp>(linalgOp.getOperation());
    if (!genericOp ||
        !(hasRecognizedEncoding(genericOp) || op->hasAttr("tri_rewrite"))) {
      return failure();
    }

    // This is pretty ugly. The idea here is to mark all encoded generic ops the
    // first time we visit, because the encodings will be erased by this
    // transformation operating on other linalg.generic ops.
    auto moduleOp = op->getParentOfType<ModuleOp>();
    moduleOp.walk([](linalg::GenericOp op) {
      if (hasRecognizedEncoding(op)) {
        op->setAttr("tri_rewrite", BoolAttr::get(op->getContext(), true));
      }
    });

    moduleOp.walk([&](Operation *op) {
      PatternRewriter::InsertionGuard insertionGuard(rewriter);
      for (auto operand : op->getOperands()) {
        eraseTriangularEncoding(operand, rewriter);
      }
    });

    auto loopRanges = linalgOp.createLoopRanges(rewriter, linalgOp.getLoc());
    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    SmallVector<Value, 4> lbs, ubs, steps;
    unpackRanges(rewriter, op->getLoc(), loopRanges, lbs, ubs, steps);
    assert(linalgOp.getNumDpsInits() > 0 &&
           "Expected at least one tensor result");
    SmallVector<Value> iterArgInitValues;
    Value zero = rewriter.create<arith::ConstantOp>(
        linalgOp.getLoc(),
        FloatAttr::get(
            linalgOp->getResultTypes()[0].cast<ShapedType>().getElementType(),
            0.0));
    for (OpOperand *outputTensor : linalgOp.getDpsInitOperands()) {
      auto outType =
          outputTensor->get().getType().dyn_cast_or_null<ShapedType>();
      assert(outType && "outType was null");
      // Perhaps a premature optimization. Using an init tensor op results in an
      // extra buffer allocation.
      Value space = rewriter.create<memref::AllocOp>(
          linalgOp.getLoc(),
          MemRefType::get(outType.getShape(), outType.getElementType()));
      if (linalgOp.payloadUsesValueFromOperand(outputTensor)) {
        rewriter.create<linalg::FillOp>(linalgOp.getLoc(), zero, space);
      }
      auto loaded =
          rewriter.create<bufferization::ToTensorOp>(linalgOp.getLoc(), space);

      iterArgInitValues.push_back(loaded);
    }
    // Fast way to get an index value of 0
    // iterArgInitValues.push_back(lbs[0]);
    auto loopNest = scf::buildLoopNest(
        rewriter, linalgOp.getLoc(), lbs, ubs, steps, iterArgInitValues,
        [&](OpBuilder &b, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> scf::ValueVector {
          auto iterNext =
              emitScalarImplementation(b, loc, ivs, iterArgs, linalgOp);
          return scf::ValueVector{iterNext.begin(), iterNext.end()};
        });

    // This is the part that modifies the loop bounds. It currently only works
    // for lower triangular loops.
    auto num_loops = loopNest.loops.size();
    auto last = loopNest.loops[num_loops - 1];
    last.setUpperBound(loopNest.loops[num_loops - 2].getInductionVar());

    op->replaceAllUsesWith(loopNest.getResults());
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

namespace {
struct TriangularLoopsPass
    : public PassWrapper<TriangularLoopsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TriangularLoopsPass)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, scf::SCFDialect>();
  }
  StringRef getArgument() const override {
    return "convert-linalg-triangular-to-loops";
  }
  StringRef getDescription() const override {
    return "Convert linalg.generic ops that operate on triangular tensors to "
           "loops.";
  }
  void runOnOperation() final {
    auto *context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    // target.addLegalOp<linalg::InitTensorOp>();
    target.addLegalOp<linalg::FillOp>();
    target.addLegalOp<linalg::YieldOp>();

    patterns.add<ConvertGenericOp>(patterns.getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::lagrad::createTriangularLoopsPass() {
  return std::make_unique<TriangularLoopsPass>();
}
