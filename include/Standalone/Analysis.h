#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
struct InsertExtractAnalysis {
  InsertExtractAnalysis(Operation *op);

  bool isPairedExtractSlice(tensor::ExtractSliceOp op) const;

  bool isPairedInsertSlice(tensor::InsertSliceOp op) const;

  tensor::InsertSliceOp getPairedInsertSlice(tensor::ExtractSliceOp op) const;

  bool isLinalgMarkedForBufferization(Operation *op) const;

private:
  DenseMap<Operation *, Operation *> extract_to_insert;
  DenseSet<Operation *> matchingInserts;
  DenseSet<Operation *> linalgInPlaceOps;

  Optional<tensor::InsertSliceOp>
  getMatchingInsertSlice(tensor::ExtractSliceOp op,
                         const DominanceInfo &dom) const;
};
} // namespace mlir
