#include "Standalone/Passes.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOpsDialect.h.inc"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
FloatType getScalarFloatType(Type operandType) {
  if (operandType.isa<FloatType>()) {
    return operandType.dyn_cast<FloatType>();
  } else if (operandType.isa<ShapedType>()) {
    auto elementType = operandType.dyn_cast<ShapedType>().getElementType();
    assert(elementType.isa<FloatType>() &&
           "Expected scalar element type to be a floating point type");
    return elementType.dyn_cast<FloatType>();
  }
  llvm::outs() << "getScalarFloatType for type " << operandType
               << " not implemented.\n";
  llvm_unreachable("");
  return nullptr;
}

class DiffOpLowering : public ConversionPattern {
public:
  explicit DiffOpLowering(MLIRContext *context)
      : ConversionPattern(standalone::DiffOp::getOperationName(), /*benefit=*/1,
                          context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto diffOp = dyn_cast<standalone::DiffOp>(op);
    auto constArgs = diffOp->getAttr("const").dyn_cast_or_null<ArrayAttr>();
    llvm::SmallDenseSet<int64_t> constSet;
    if (constArgs) {
      for (auto attr : constArgs.getAsValueRange<IntegerAttr>()) {
        constSet.insert(attr.getSExtValue());
      }
    }

    // Obtain a SymbolRefAttr to the Enzyme autodiff function.
    // TODO: Only works for one unique differentiated function type at a time.
    auto primalType = op->getOperand(0).getType().dyn_cast<FunctionType>();
    auto floatType = getScalarFloatType(primalType.getInput(0));
    auto llvmFloatPtr = LLVM::LLVMPointerType::get(floatType);
    assert(primalType.getNumResults() < 2 &&
           "Expected 0 or 1 results from the primal");

    auto sym = getOrInsertAutodiffDecl(rewriter, op, floatType);
    auto const_global = getOrInsertEnzymeConstDecl(rewriter, op);
    auto llvmI32PtrTy =
        LLVM::LLVMPointerType::get(IntegerType::get(op->getContext(), 32));
    auto enzyme_const_addr = rewriter.create<LLVM::AddressOfOp>(
        op->getLoc(), llvmI32PtrTy, const_global);

    // Following the autograd-style API, we want to take the result of the diff
    // operator and replace it with a call to the Enzyme API
    auto result = op->getResult(0);
    rewriter.eraseOp(op);

    // TODO: How are we supposed to populate the type converter member?
    LLVMTypeConverter typeConverter(op->getContext());

    for (auto it = result.use_begin(); it != result.use_end(); it++) {
      auto user = dyn_cast_or_null<CallIndirectOp>(it.getUser());
      assert(user && "Expected user to be a CallIndirectOp");
      // Copy over the arguments for the op
      auto arguments = SmallVector<Value>();
      arguments.push_back(operands[0]);

      auto opIt = user->getOperands().drop_front(1);
      size_t arg_index = 0;
      for (auto it = opIt.begin(); it != opIt.end(); it++) {
        // auto m = *arg;
        auto arg = *it;
        auto memrefType = arg.getType().dyn_cast_or_null<MemRefType>();
        if (constSet.contains(arg_index)) {
          assert(memrefType && "Operator marked const was not a MemRef");
        }
        if (memrefType) {
          auto rank = memrefType.getRank();
          // Ignore the first pointer
          arguments.push_back(enzyme_const_addr.getResult());
          auto casted = rewriter
                            .create<UnrealizedConversionCastOp>(
                                user->getLoc(),
                                typeConverter.convertType(arg.getType()), arg)
                            .getResult(0);
          arguments.push_back(rewriter
                                  .create<LLVM::ExtractValueOp>(
                                      user->getLoc(), llvmFloatPtr, casted,
                                      rewriter.getI64ArrayAttr(0))
                                  .getResult());

          if (constSet.contains(arg_index)) {
            // The aligned pointer must be marked const
            arguments.push_back(enzyme_const_addr.getResult());
          }
          arguments.push_back(rewriter
                                  .create<LLVM::ExtractValueOp>(
                                      user->getLoc(), llvmFloatPtr, casted,
                                      rewriter.getI64ArrayAttr(1))
                                  .getResult());

          if (!constSet.contains(arg_index)) {
            // Shadow pointer has to follow the aligned pointer
            auto shadow = *(++it);
            assert(shadow && shadow.getType().isa<MemRefType>() &&
                   "Shadow argument must be a Memref");
            auto shadowCasted =
                rewriter
                    .create<UnrealizedConversionCastOp>(
                        shadow.getLoc(),
                        typeConverter.convertType(shadow.getType()), shadow)
                    .getResult(0);
            auto extractShadowOp = rewriter.create<LLVM::ExtractValueOp>(
                shadow.getLoc(), llvmFloatPtr, shadowCasted,
                rewriter.getI64ArrayAttr(1));
            arguments.push_back(extractShadowOp.getResult());
          }

          auto llvmI64Ty = IntegerType::get(user->getContext(), 64);
          arguments.push_back(rewriter
                                  .create<LLVM::ExtractValueOp>(
                                      user->getLoc(), llvmI64Ty, casted,
                                      rewriter.getI64ArrayAttr(2))
                                  .getResult());
          for (int64_t i = 0; i < rank; ++i) {
            arguments.push_back(rewriter
                                    .create<LLVM::ExtractValueOp>(
                                        user->getLoc(), llvmI64Ty, casted,
                                        rewriter.getI64ArrayAttr({3, i}))
                                    .getResult());
          }
          for (int64_t i = 0; i < rank; ++i) {
            arguments.push_back(rewriter
                                    .create<LLVM::ExtractValueOp>(
                                        user->getLoc(), llvmI64Ty, casted,
                                        rewriter.getI64ArrayAttr({4, i}))
                                    .getResult());
          }
        } else {
          arguments.push_back(arg);
        }
        arg_index++;
      }

      rewriter.replaceOpWithNewOp<CallOp>(user, sym, primalType.getResults(),
                                          arguments);
    }

    return success();
  }

private:
  static FlatSymbolRefAttr getOrInsertEnzymeConstDecl(PatternRewriter &rewriter,
                                                      Operation *op) {
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    auto *context = moduleOp.getContext();
    if (moduleOp.lookupSymbol<LLVM::GlobalOp>("enzyme_const")) {
      return SymbolRefAttr::get(context, "enzyme_const");
    }

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto llvmI32Ty = IntegerType::get(context, 32);
    rewriter.create<LLVM::GlobalOp>(moduleOp.getLoc(), llvmI32Ty,
                                    /*isConstant=*/true,
                                    LLVM::Linkage::Linkonce, "enzyme_const",
                                    IntegerAttr::get(llvmI32Ty, 0));
    return SymbolRefAttr::get(context, "enzyme_const");
  }

  static FlatSymbolRefAttr getOrInsertAutodiffDecl(PatternRewriter &rewriter,
                                                   Operation *op,
                                                   FloatType returnType) {
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    auto *context = moduleOp.getContext();
    if (moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("__enzyme_autodiff")) {
      return SymbolRefAttr::get(context, "__enzyme_autodiff");
    }

    // Create the function declaration for __enzyme_autodiff
    LLVMTypeConverter typeConverter(context);
    auto llvmOriginalFuncType =
        typeConverter.packFunctionResults(op->getOperand(0).getType());

    auto llvmFnType = LLVM::LLVMFunctionType::get(
        returnType, llvmOriginalFuncType, /*isVarArg=*/true);

    // Insert the autodiff function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(moduleOp.getLoc(), "__enzyme_autodiff",
                                      llvmFnType);
    return SymbolRefAttr::get(context, "__enzyme_autodiff");
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
  StringRef getArgument() const override {
    return "convert-standalone-to-llvm";
  }
  StringRef getDescription() const override {
    return "Lower standalone.diff calls to Enzyme compatible calls";
  }
  void runOnOperation() final;
};
} // end anonymous namespace

void StandaloneToLLVMLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  target.addIllegalOp<standalone::DiffOp>();

  LLVMTypeConverter typeConverter(&getContext());

  OwningRewritePatternList patterns(&getContext());
  populateLoopToStdConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);
  patterns.insert<DiffOpLowering>(&getContext());

  auto mod = getOperation();
  if (failed(applyPartialConversion(mod, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::Standalone::createLowerToLLVMPass() {
  return std::make_unique<StandaloneToLLVMLoweringPass>();
}
