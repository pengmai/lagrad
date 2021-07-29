#include <memory>

namespace mlir {
class Pass;

namespace Standalone {
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

std::unique_ptr<mlir::Pass> createGradPass();

std::unique_ptr<mlir::Pass> createTensorConstantFoldPass();

std::unique_ptr<mlir::Pass> createElementwiseToAffinePass();
} // end namespace Standalone
} // end namespace mlir