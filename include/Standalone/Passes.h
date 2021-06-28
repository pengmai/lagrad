#include <memory>

namespace mlir {
class Pass;

namespace Standalone {
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

std::unique_ptr<mlir::Pass> createGradPass();

} // end namespace Standalone
} // end namespace mlir