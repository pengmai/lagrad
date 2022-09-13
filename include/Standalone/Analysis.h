#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include <string>

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

enum class HotSparsityType { Empty, OneHot, RowHot, ColHot };

struct SparsePropagation {
  SparsePropagation(Operation *op);
  Optional<HotSparsityType> getSparsityType(Value val);
  Optional<HotSparsityType> getSparsityEncoding(RankedTensorType type);
  void setIndices(Value tensor, Value indices);
  Optional<Value> getIndices(Value tensor);

private:
  DenseMap<Value, std::string> debug_names;
  DenseMap<Value, HotSparsityType> sparsityTypes;
  DenseMap<Value, Value> indices;
  void propagateInsertSlice(tensor::InsertSliceOp op);
  void propagateLinalgGeneric(linalg::GenericOp op);
};
} // namespace mlir
