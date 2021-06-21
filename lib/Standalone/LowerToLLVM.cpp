#include "Standalone/Passes.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOpsDialect.h.inc"
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
    auto const_global = getOrInsertEnzymeConstDecl(rewriter, op);
    auto llvmI32PtrTy =
        LLVM::LLVMPointerType::get(IntegerType::get(op->getContext(), 32));
    auto enzyme_const_addr = rewriter.create<LLVM::AddressOfOp>(
        op->getLoc(), llvmI32PtrTy, const_global);

    // Following the autograd-style API, we want to take the result of the diff
    // operator and replace it with a call to the Enzyme API
    auto result = op->getResult(0);
    rewriter.eraseOp(op);
    for (auto it = result.use_begin(); it != result.use_end(); it++) {
      // TODO: We currently assume that user is a call_indirect op.
      auto user = it.getUser();
      // Copy over the arguments for the op
      auto arguments = std::vector<mlir::Value>();
      arguments.push_back(operands[0]);
      // TODO: Need to actually check the types here
      auto llvmF32Ptr =
          LLVM::LLVMPointerType::get(FloatType::getF32(user->getContext()));
      auto opIt = user->getOperands().drop_front(1);
      for (auto it = opIt.begin(); it != opIt.end(); it++) {
        // auto m = *arg;
        auto arg = *it;
        if (arg.getType().isa<MemRefType>()) {
          // Ignore the first pointer
          arguments.push_back(enzyme_const_addr.getResult());
          arguments.push_back(
              rewriter
                  .create<LLVM::ExtractValueOp>(user->getLoc(), llvmF32Ptr, arg,
                                                rewriter.getI64ArrayAttr(0))
                  .getResult());
          arguments.push_back(
              rewriter
                  .create<LLVM::ExtractValueOp>(user->getLoc(), llvmF32Ptr, arg,
                                                rewriter.getI64ArrayAttr(1))
                  .getResult());

          // Shadow pointer has to follow the aligned pointer
          auto shadow = *(++it);
          assert(shadow.getType().isa<MemRefType>() &&
                 "Shadow argument must be a Memref");
          auto extractShadowOp = rewriter.create<LLVM::ExtractValueOp>(
              shadow.getLoc(), llvmF32Ptr, shadow, rewriter.getI64ArrayAttr(1));
          arguments.push_back(extractShadowOp.getResult());

          auto llvmI64Ty = IntegerType::get(user->getContext(), 64);
          arguments.push_back(
              rewriter
                  .create<LLVM::ExtractValueOp>(user->getLoc(), llvmI64Ty, arg,
                                                rewriter.getI64ArrayAttr(2))
                  .getResult());
          arguments.push_back(rewriter
                                  .create<LLVM::ExtractValueOp>(
                                      user->getLoc(), llvmI64Ty, arg,
                                      rewriter.getI64ArrayAttr({3, 0}))
                                  .getResult());
          arguments.push_back(rewriter
                                  .create<LLVM::ExtractValueOp>(
                                      user->getLoc(), llvmI64Ty, arg,
                                      rewriter.getI64ArrayAttr({4, 0}))
                                  .getResult());
        }
      }

      rewriter.replaceOpWithNewOp<mlir::CallOp>(
          user, sym, rewriter.getF32Type(), ArrayRef<Value>(arguments));
    }

    return success();
  }

private:
  static FlatSymbolRefAttr getOrInsertEnzymeConstDecl(PatternRewriter &rewriter,
                                                      Operation *op) {
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    auto *context = moduleOp.getContext();
    // if (moduleOp.lookupSymbol<LLVM::GlobalOp>("enzyme_const")) {
    //   return SymbolRefAttr::get("enzyme_const", context);
    // }

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto llvmI32Ty = IntegerType::get(context, 32);
    rewriter.create<LLVM::GlobalOp>(
        moduleOp.getLoc(), llvmI32Ty, /*isConstant=*/true,
        LLVM::Linkage::External, "enzyme_const", nullptr);
    // rewriter.setInsertionPointAfter(op);
    // rewriter.create<LLVM::AddressOfOp>(op->getLoc(), globalOp);
    return SymbolRefAttr::get("enzyme_const", context);
  }

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
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  patterns.insert<DiffOpLowering>(&getContext());

  auto mod = getOperation();
  if (failed(applyFullConversion(mod, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::Standalone::createLowerToLLVMPass() {
  return std::make_unique<StandaloneToLLVMLoweringPass>();
}
