#include <memory>

namespace mlir {
class Pass;

namespace Standalone {
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

std::unique_ptr<mlir::Pass> createGradPass();

std::unique_ptr<mlir::Pass> createTensorConstantFoldPass();

std::unique_ptr<mlir::Pass> createElementwiseToAffinePass();

std::unique_ptr<mlir::Pass> createBufferizePass();

std::unique_ptr<mlir::Pass> createTriangularLoopsPass();

std::unique_ptr<mlir::Pass> createStaticAllocsPass();

std::unique_ptr<mlir::Pass> createStandaloneDCEPass();
} // end namespace Standalone
} // end namespace mlir