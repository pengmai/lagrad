#include "LAGrad/Analysis.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"

namespace mlir {

constexpr bool disable_ie_analysis = false;
InsertExtractAnalysis::InsertExtractAnalysis(Operation *op) {
  if (disable_ie_analysis) {
    return;
  }

  DominanceInfo dom;
  op->walk([&](tensor::ExtractSliceOp op) {
    auto maybeInsertSliceOp = getMatchingInsertSlice(op, dom);
    if (maybeInsertSliceOp.hasValue()) {
      extract_to_insert[op] = maybeInsertSliceOp.getValue();
      matchingInserts.insert(maybeInsertSliceOp.getValue());
      for (auto &use : op.getResult().getUses()) {
        if (auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner())) {
          // Only handle the case where the linalg op has one output for now.
          if (linalgOp.isOutputTensor(&use) && linalgOp.getNumOutputs() == 1 &&
              linalgOp.hasTensorSemantics()) {
            linalgInPlaceOps.insert(linalgOp);
          }
        }
      }
    }
  });
}

bool InsertExtractAnalysis::isPairedExtractSlice(
    tensor::ExtractSliceOp op) const {
  return extract_to_insert.count(op) > 0;
}

bool InsertExtractAnalysis::isPairedInsertSlice(
    tensor::InsertSliceOp op) const {
  return matchingInserts.contains(op);
}

tensor::InsertSliceOp
InsertExtractAnalysis::getPairedInsertSlice(tensor::ExtractSliceOp op) const {
  return cast<tensor::InsertSliceOp>(extract_to_insert.lookup(op));
}

bool InsertExtractAnalysis::isLinalgMarkedForBufferization(
    Operation *op) const {
  return linalgInPlaceOps.count(op) > 0;
}

Optional<tensor::InsertSliceOp>
InsertExtractAnalysis::getMatchingInsertSlice(tensor::ExtractSliceOp op,
                                              const DominanceInfo &dom) const {
  // The source of the extract slice should have exactly one use besides the
  // insert slice op.
  size_t domUseCount = 0;
  tensor::InsertSliceOp insertSliceOp;
  for (Operation *user : op.source().getUsers()) {
    if (dom.properlyDominates(op.getResult(), user)) {
      if (auto iso = dyn_cast<tensor::InsertSliceOp>(user)) {
        if (iso.dest() == op.source()) {
          insertSliceOp = iso;
        }
      }
      domUseCount++;
    }
  }
  if (domUseCount != 1) {
    return llvm::None;
  }

  // We ultimately remove the insertSliceOp, so we must ensure that the
  // subsequent ops write in-place.
  bool isWrittenInPlace = false;
  for (OpOperand &use : op.getResult().getUses()) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner())) {
      if (linalgOp.isOutputTensor(&use)) {
        isWrittenInPlace = true;
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(use.getOwner())) {
      // This is potentially a dangerous assumption.
      isWrittenInPlace = true;
    } else if (auto addfOp = dyn_cast<arith::AddFOp>(use.getOwner())) {
      isWrittenInPlace = true;
    }
  }

  bool linked = false;
  if (insertSliceOp) {
    SmallVector<Value> frontier{op.getResult()};
    ValueSet derivedFromResult;
    runTopDownDFS(frontier, derivedFromResult);
    linked = derivedFromResult.contains(insertSliceOp.source());
  }

  if (linked && isWrittenInPlace && insertSliceOp &&
      (insertSliceOp.getMixedOffsets() == op.getMixedOffsets()) &&
      (insertSliceOp.getMixedSizes() == op.getMixedSizes()) &&
      (insertSliceOp.getMixedStrides() == op.getMixedStrides())) {
    return insertSliceOp;
  }
  return llvm::None;
}
} // namespace mlir
