#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/AnalysisManager.h"
#include <string>

namespace mlir {
struct InsertExtractAnalysis {
  InsertExtractAnalysis(Operation *op);

  void disableAnalysis();

  bool isPairedExtractSlice(tensor::ExtractSliceOp op) const;

  bool isPairedInsertSlice(tensor::InsertSliceOp op) const;

  tensor::InsertSliceOp getPairedInsertSlice(tensor::ExtractSliceOp op) const;

  bool isLinalgMarkedForBufferization(Operation *op) const;

private:
  bool disabled;
  DenseMap<Operation *, Operation *> extract_to_insert;
  DenseSet<Operation *> matchingInserts;
  DenseSet<Operation *> linalgInPlaceOps;

  Optional<tensor::InsertSliceOp>
  getMatchingInsertSlice(tensor::ExtractSliceOp op,
                         const DominanceInfo &dom) const;
};

enum class HotSparsityType { Empty, OneHot, RowHot, ColHot };

struct LoopNest {
  SmallVector<scf::ForOp> loops;
  SmallVector<Value> inductionVars, inputTensorOperands, inputRegionArgs,
      outputTensorOperands, outputRegionArgs, results;
  SmallVector<AffineMap> inputMaps;
  SmallVector<unsigned> outputPerIterWrites;
  DenseSet<Operation *> ivComputation;
};

struct SparsePropagation {
  SparsePropagation(Operation *op, AnalysisManager &am);
  Optional<HotSparsityType> getSparsityType(Value val) const;
  Optional<HotSparsityType> getSparsityEncoding(RankedTensorType type) const;
  void setIndices(Value tensor, Value indices);
  Optional<Value> getIndices(Value tensor);

private:
  DenseMap<Value, std::string> debug_names;
  DenseMap<Value, HotSparsityType> sparsityTypes;
  DenseMap<Value, Value> indices;
  void propagateInsertSlice(tensor::InsertSliceOp op);
  void propagateLinalgGeneric(linalg::GenericOp op);
  void propagateSCFFor(scf::ForOp op);
  void propagateLoopNest(LoopNest loopNest);
};

struct LoopNestAnalysis {
public:
  LoopNestAnalysis(Operation *op);
  Optional<LoopNest> getLoopNest(scf::ForOp op) const;
  bool isLoopNest(scf::ForOp op) const;

private:
  DenseMap<Operation *, LoopNest> forOpMapping;
};
} // namespace mlir
