#include "LAGrad/Passes.h"
#include "LAGrad/LAGradDialect.h"
#include "LAGrad/LAGradOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

// TODO: What's the best way to incorporate this?
namespace {
class TensorConstantFolding : public ConversionPattern {
public:
  explicit TensorConstantFolding(MLIRContext *context)
      : ConversionPattern(lagrad::GradOp::getOperationName(), /*benefit=*/1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return success();
  }
};
} // namespace