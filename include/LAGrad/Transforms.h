#pragma once
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace lagrad {

void populateLAGradTransforms(RewritePatternSet &patterns,
                              MLIRContext *ctx);

} // end namespace lagrad
} // end namespace mlir
