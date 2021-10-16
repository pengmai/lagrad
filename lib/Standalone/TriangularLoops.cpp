/**
 * A pass to lower linalg.generic ops that operate on triangular operands to
 * loops.
 */
#include "Standalone/Passes.h"
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

namespace {
// bool hasRecognizedEncoding(linalg::GenericOp op) {
//   for (auto operand : op.getOperands()) {
//     auto encoding =
//         operand.getType().dyn_cast<RankedTensorType>().getEncoding();
//     auto val = encoding.dyn_cast_or_null<StringAttr>();
//     if (val && val.getValue() == "ltri") {
//       return true;
//     }
//   }
//   return false;
// }

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
  assert(iterArgs.size() == linalgOp.getOutputTensorOperands().size() &&
         "Expected # of iter args to be equal to # of output tensor operands.");

  SmallVector<Value> indexedValues;
  indexedValues.reserve(linalgOp.getNumInputsAndOutputs());

  auto allIvsPlusDims = SmallVector<Value>(allIvs.begin(), allIvs.end());
  // 1.a. Emit load from input operands or for scalars access the operand
  // itself.
  for (auto inputOperand : linalgOp.getInputOperands()) {
    if (linalgOp.isScalar(inputOperand)) {
      indexedValues.push_back(inputOperand->get());
      continue;
    }
    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getTiedIndexingMap(inputOperand), allIvsPlusDims);
    indexedValues.push_back(
        b.create<tensor::ExtractOp>(loc, inputOperand->get(), indexing));
  }

  // 1.b. Emit load from output views.
  for (auto pair : llvm::zip(linalgOp.getOutputOperands(), iterArgs)) {
    auto outputOperand = std::get<0>(pair);
    Value iterArg = std::get<1>(pair);

    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getTiedIndexingMap(outputOperand), allIvsPlusDims);
    indexedValues.push_back(
        b.create<tensor::ExtractOp>(loc, iterArg, indexing));
  }

  // 2. Inline region, currently only works for a single basic block.
  // 3. Emit store.
  SmallVector<Value> outputTensors{iterArgs};
  SmallVector<SmallVector<Value>, 8> indexing;
  for (auto outputOperand : linalgOp.getOutputTensorOperands()) {
    indexing.push_back(makeCanonicalAffineApplies(
        b, loc, linalgOp.getTiedIndexingMap(outputOperand), allIvsPlusDims));
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
    if (!linalgOp) {
      return failure();
    }
    if (!linalgOp.hasTensorSemantics()) {
      return failure();
    }
    auto genericOp =
        dyn_cast_or_null<linalg::GenericOp>(linalgOp.getOperation());
    if (!genericOp) {
      return failure();
    }
    // if (!hasRecognizedEncoding(genericOp)) {
    //   return failure();
    // }

    auto loopRanges = linalgOp.createLoopRanges(rewriter, linalgOp.getLoc());
    auto iteratorTypes =
        llvm::to_vector<4>(linalgOp.iterator_types().getValue());
    SmallVector<Value, 4> lbs, ubs, steps;
    unpackRanges(loopRanges, lbs, ubs, steps);
    SmallVector<Value> iterArgInitValues = linalgOp.getOutputTensorOperands();
    auto loopNest = scf::buildLoopNest(
        rewriter, linalgOp.getLoc(), lbs, ubs, steps, iterArgInitValues,
        [&](OpBuilder &b, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> scf::ValueVector {
          auto iterNext =
              emitScalarImplementation(b, loc, ivs, iterArgs, linalgOp);
          return scf::ValueVector{iterNext.begin(), iterNext.end()};
        });
    op->replaceAllUsesWith(
        llvm::makeArrayRef(linalgOp.getOutputOperand(0)->get()));
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

namespace {
struct TriangularLoopsPass
    : public PassWrapper<TriangularLoopsPass, OperationPass<ModuleOp>> {
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
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    // target.addIllegalOp<linalg::GenericOp>();

    patterns.add<ConvertGenericOp>(patterns.getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::Standalone::createTriangularLoopsPass() {
  return std::make_unique<TriangularLoopsPass>();
}
