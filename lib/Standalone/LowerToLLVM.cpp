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

inline void print(llvm::StringRef str) { llvm::outs() << str; }

namespace {
class DiffOpLowering : public ConversionPattern {
public:
  explicit DiffOpLowering(MLIRContext *context)
      : ConversionPattern(standalone::DiffOp::getOperationName(), /*benefit=*/1,
                          context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    print("\n---BEGIN---\n");
    print("The operation:\n");
    op->print(llvm::outs());
    print("\n");

    // Obtain a SymbolRefAttr to the Enzyme external function.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    parentModule.lookupSymbol<LLVM::LLVMFuncOp>("__enzyme_autodiff");
    auto sym = SymbolRefAttr::get("__enzyme_autodiff", op->getContext());

    // Following the autograd-style API, we want to take the result of the diff
    // operator and replace it with a call to the Enzyme API
    auto result = op->getResult(0);
    rewriter.eraseOp(op);
    for (auto it = result.use_begin(); it != result.use_end(); it++) {
      print("The user:\n");
      it.getUser()->print(llvm::outs());
      auto user = it.getUser();
      // Hardcoded (f32) -> f32 signature right now
      rewriter.replaceOpWithNewOp<mlir::CallOp>(
          user, sym, rewriter.getF32Type(),
          ArrayRef<Value>({op->getOperand(0), user->getOperand(1)}));
      print("\n");
    }
    print("\n---END---\n");
    // op->getResult(0).
    // op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    // rewriter.eraseOp(op);
    return success();
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
