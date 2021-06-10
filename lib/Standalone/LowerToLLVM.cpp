#include "Standalone/Passes.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
class DiffOpLowering : public ConversionPattern {
public:
  explicit DiffOpLowering(MLIRContext *context)
      : ConversionPattern(standalone::DiffOp::getOperationName(), /*benefit=*/1,
                          context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Obtain a SymbolRefAttr to the Enzyme autodiff function.
    // TODO: Only works for one unique differentiated function type at a time.
    auto sym = getOrInsertAutodiffDecl(rewriter, op);

    // Following the autograd-style API, we want to take the result of the diff
    // operator and replace it with a call to the Enzyme API
    auto result = op->getResult(0);
    rewriter.eraseOp(op);
    for (auto it = result.use_begin(); it != result.use_end(); it++) {
      auto user = it.getUser();
      // Copy over the arguments for the op
      auto arguments = std::vector<mlir::Value>();
      arguments.push_back(op->getOperand(0));
      for (Value arg : user->getOperands().drop_front(1)) {
        arguments.push_back(arg);
      }

      rewriter.replaceOpWithNewOp<mlir::CallOp>(
          user, sym, rewriter.getF32Type(), ArrayRef<Value>(arguments));
    }

    return success();
  }

private:
  static FlatSymbolRefAttr getOrInsertAutodiffDecl(PatternRewriter &rewriter,
                                                   Operation *op) {
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    auto *context = moduleOp.getContext();
    if (moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("__enzyme_autodiff")) {
      return SymbolRefAttr::get("__enzyme_autodiff", context);
    }

    // Create the function declaration for __enzyme_autodiff
    auto llvmF32Ty = FloatType::getF32(context);

    LLVMTypeConverter typeConverter(context);
    auto llvmOriginalFuncType =
        typeConverter.packFunctionResults(op->getOperand(0).getType());

    auto llvmFnType = LLVM::LLVMFunctionType::get(
        llvmF32Ty, llvmOriginalFuncType, /*isVarArg=*/true);

    // Insert the autodiff function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(moduleOp.getLoc(), "__enzyme_autodiff",
                                      llvmFnType);
    return SymbolRefAttr::get("__enzyme_autodiff", context);
  }
};
} // end anonymous namespace

namespace {
struct StandaloneToLLVMLoweringPass
    : public PassWrapper<StandaloneToLLVMLoweringPass,
                         OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() final;
};
} // end anonymous namespace

void StandaloneToLLVMLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  LLVMTypeConverter typeConverter(&getContext());

  OwningRewritePatternList patterns;
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  patterns.insert<DiffOpLowering>(&getContext());

  auto mod = getOperation();
  if (failed(applyFullConversion(mod, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::Standalone::createLowerToLLVMPass() {
  return std::make_unique<StandaloneToLLVMLoweringPass>();
}
