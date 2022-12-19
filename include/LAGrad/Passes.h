#include <memory>

namespace mlir {
class Pass;

namespace lagrad {
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

std::unique_ptr<mlir::Pass> createGradPass();

std::unique_ptr<mlir::Pass> createTensorConstantFoldPass();

std::unique_ptr<mlir::Pass> createBufferizePass();

std::unique_ptr<mlir::Pass> createTriangularLoopsPass();

std::unique_ptr<mlir::Pass> createPackTriangularPass();

std::unique_ptr<mlir::Pass> createStaticAllocsPass();

std::unique_ptr<mlir::Pass> createStandaloneDCEPass();

std::unique_ptr<mlir::Pass> createLoopHoistingPass();

std::unique_ptr<mlir::Pass> createLinalgCanonicalizePass();

std::unique_ptr<mlir::Pass> createLinalgToKnownLibraryCallPass();

std::unique_ptr<mlir::Pass> createSparsifyPass();
} // namespace lagrad
} // end namespace mlir