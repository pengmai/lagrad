#include <memory>

namespace mlir {
class Pass;

namespace Standalone {
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // end namespace Standalone
} // end namespace mlir