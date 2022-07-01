#include "Standalone/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
using namespace mlir;
void populatePrimalCaches(LAGradContext &ctx, FuncOp primalFunc,
                          ConversionPatternRewriter &rewriter) {
  // llvm::errs() << "Running populate primal caches\n";
  // primalFunc.walk([&](scf::ForOp forOp) {
  //   llvm::errs() << "found for op: " << forOp << "\n\n";
  // });
}
} // namespace mlir
