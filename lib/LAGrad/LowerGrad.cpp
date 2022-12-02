#include "LAGrad/LAGradDialect.h"
#include "LAGrad/LAGradOps.h"
#include "LAGrad/Passes.h"
#include "LAGrad/Transforms.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
class GradOpLowering : public OpConversionPattern<lagrad::GradOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(lagrad::GradOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = generateAdjointFunc(op, rewriter);
    if (!funcOp) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, funcOp, op.getOperands());
    return success();
  }

private:
  static func::FuncOp generateAdjointFunc(lagrad::GradOp gradOp,
                                          ConversionPatternRewriter &rewriter) {
    auto moduleOp = gradOp->getParentOfType<ModuleOp>();
    auto originalFuncOp =
        moduleOp.lookupSymbol<func::FuncOp>(gradOp.getFAttr());

    std::string adjointFuncName = ("__grad_" + originalFuncOp.getName()).str();

    if (auto existingAdjoint =
            moduleOp.lookupSymbol<func::FuncOp>(adjointFuncName)) {
      return existingAdjoint;
    }

    // If we differentiate the same function multiple times w.r.t. different
    // args this will fail. Handling this properly requires some kind of name
    // mangling.
    auto gradientsOf = gradOp->getAttr("of").dyn_cast_or_null<ArrayAttr>();
    bool customGradSignal = gradOp->hasAttrOfType<UnitAttr>("grad_signal");
    bool oneHotSparse = gradOp->hasAttrOfType<UnitAttr>("sparse");

    // If we received request for a custom gradient signal, this is equivalent
    // to taking in the gradient signal as a parameter.
    func::FuncOp funcOp =
        copyFunctionDeclaration(originalFuncOp, adjointFuncName, rewriter);
    LAGradContext lagradctx{moduleOp};
    DEBUGpopulateFunc(lagradctx.debug_names, funcOp);
    analyzeDynamicShapes(lagradctx, funcOp, rewriter);
    runActivityAnalysis(lagradctx, funcOp, gradientsOf);
    populatePrimalCaches(lagradctx, funcOp, rewriter);
    return differentiateFunction(funcOp, lagradctx, gradientsOf, rewriter,
                                 /*topLevel=*/!customGradSignal,
                                 /*onehotsparse=*/oneHotSparse);
  }
};
} // end anonymous namespace

namespace {
struct GradTarget : public ConversionTarget {
  GradTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<bufferization::BufferizationDialect>();
    addLegalDialect<math::MathDialect>();
    addLegalDialect<memref::MemRefDialect>();
    addLegalDialect<scf::SCFDialect>();
    addLegalDialect<tensor::TensorDialect>();
    addLegalDialect<linalg::LinalgDialect>();
    addLegalDialect<func::FuncDialect>();
    addIllegalDialect<lagrad::LAGradDialect>();
    addLegalOp<lagrad::PackOp>();
  }
};
} // end anonymous namespace

namespace {
struct GradConversionPass
    : public PassWrapper<GradConversionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GradConversionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect,
                    bufferization::BufferizationDialect>();
  }
  StringRef getArgument() const override { return "take-grads"; }
  StringRef getDescription() const override {
    return "Run the autodiff procedure for lagrad.grad";
  }
  void runOnOperation() final;
};
} // end anonymous namespace

void GradConversionPass::runOnOperation() {
  GradTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  RewritePatternSet patterns(&getContext());
  patterns.insert<GradOpLowering>(&getContext());
  lagrad::populateLAGradTransforms(patterns, &getContext());

  auto mod = getOperation();
  if (failed(applyFullConversion(mod, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::lagrad::createGradPass() {
  return std::make_unique<GradConversionPass>();
}
