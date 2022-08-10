#include "Standalone/Logger.h"
#include "Standalone/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using llvm::errs;
namespace mlir {
void analyzeDynamicShapes(LAGradContext &ctx, FuncOp funcOp,
                          OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&funcOp.getBody().getBlocks().front());
  auto intToAttr = [&](int64_t i) {
    return IntegerAttr::get(IntegerType::get(builder.getContext(), 64), i);
  };
  for (BlockArgument arg : funcOp.getArguments()) {
    if (auto type = arg.getType().dyn_cast<RankedTensorType>()) {
      if (type.getNumDynamicDims() > 0) {
        SmallVector<OpFoldResult, 4> shape;
        shape.reserve(type.getRank());
        for (int64_t idx = 0; idx < type.getRank(); idx++) {
          if (type.isDynamicDim(idx)) {
            shape.push_back(
                builder.create<tensor::DimOp>(arg.getLoc(), arg, idx)
                    .getResult());
          } else {
            shape.push_back(intToAttr(type.getDimSize(idx)));
          }
        }
        ctx.dynamic_shapes.insert(std::make_pair(arg, shape));
      }
    }
  }

  funcOp.walk([&](linalg::InitTensorOp op) {
    RankedTensorType type = op.getType();
    if (type.getNumDynamicDims() > 0) {
      SmallVector<OpFoldResult> shape;
      shape.reserve(type.getRank());
      for (int64_t idx = 0; idx < type.getRank(); idx++) {
        if (type.isDynamicDim(idx)) {
          shape.push_back(op.getDynamicSize(idx));
        } else {
          shape.push_back(intToAttr(op.getStaticSize(idx)));
        }
      }
      ctx.dynamic_shapes.insert(std::make_pair(op.getResult(), shape));
    }
  });

  funcOp.walk([&](linalg::LinalgOp op) {
    if (isa<linalg::InitTensorOp>(op)) {
      return;
    }
    for (OpOperand *operand : op.getOutputTensorOperands()) {
      if (auto type = operand->get().getType().cast<RankedTensorType>()) {
        if (type.getNumDynamicDims() > 0) {
          assert(ctx.dynamic_shapes.count(operand->get()) &&
                 "linalg op had dynamic shape but was not in the dynamic "
                 "shape map");
          ctx.dynamic_shapes.insert(
              std::make_pair(op.getTiedOpResult(operand),
                             ctx.dynamic_shapes.lookup(operand->get())));
        }
      }
    }
  });
}
} // namespace mlir
