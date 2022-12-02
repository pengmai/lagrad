#include "LAGrad/Passes.h"
#include "LAGrad/LAGradDialect.h"
#include "LAGrad/LAGradOps.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
      : ConversionPattern(lagrad::DiffOp::getOperationName(), /*benefit=*/1,
                          context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto diffOp = dyn_cast<lagrad::DiffOp>(op);
    auto constArgs = diffOp->getAttr("const").dyn_cast_or_null<ArrayAttr>();
    llvm::SmallDenseSet<int64_t> constSet;
    if (constArgs) {
      for (auto attr : constArgs.getAsValueRange<IntegerAttr>()) {
        constSet.insert(attr.getSExtValue());
      }
    }

    // Obtain a SymbolRefAttr to the Enzyme autodiff function.
    // TODO: Only works for one unique differentiated function type at a
    // time.
    auto primalType = op->getOperand(0).getType().dyn_cast<FunctionType>();
    auto floatType = getScalarFloatType(primalType.getInput(0));
    assert(primalType.getNumResults() < 2 &&
           "Expected 0 or 1 results from the primal");

    auto sym = getOrInsertAutodiffDecl(rewriter, op, floatType);
    auto const_global = getOrInsertEnzymeConstDecl(rewriter, op);
    auto llvmI8PtrTy =
        LLVM::LLVMPointerType::get(IntegerType::get(op->getContext(), 8));
    auto enzyme_const_addr = rewriter.create<LLVM::AddressOfOp>(
        op->getLoc(), llvmI8PtrTy, const_global);

    // Following the autograd-style API, we want to take the result of the diff
    // operator and replace it with a call to the Enzyme API
    auto result = op->getResult(0);
    rewriter.eraseOp(op);

    // TODO: How are we supposed to populate the type converter member?
    LLVMTypeConverter typeConverter(op->getContext());
    Location loc = op->getLoc();

    for (auto it = result.use_begin(); it != result.use_end(); it++) {
      auto user = dyn_cast_or_null<func::CallIndirectOp>(it.getUser());
      assert(user && "Expected user to be a CallIndirectOp");
      // Copy over the arguments for the op
      auto arguments = SmallVector<Value>();
      using llvm::errs;
      errs() << "first operand: " << operands[0] << "\n";
      errs() << "callable for callee: "
             << user.getCallableForCallee().get<Value>() << "\n";
      Value llvmPrimalFunc =
          rewriter
              .create<UnrealizedConversionCastOp>(
                  op->getLoc(),
                  typeConverter.convertType(operands[0].getType()), operands[0])
              .getResult(0);
      auto castedPrimal =
          rewriter.create<LLVM::BitcastOp>(loc, llvmI8PtrTy, llvmPrimalFunc);
      // auto castedPrimal = rewriter.create<LLVM::AddressOfOp>(
      //     op->getLoc(), llvmI8PtrTy, primalConst.getValueAttr());
      arguments.push_back(castedPrimal);

      auto opIt = user->getOperands().drop_front(1);
      size_t arg_index = 0;
      for (auto it = opIt.begin(); it != opIt.end(); it++) {
        // auto m = *arg;
        Value arg = *it;
        auto memrefType = arg.getType().dyn_cast_or_null<MemRefType>();
        if (constSet.contains(arg_index)) {
          assert(memrefType && "Operator marked const was not a MemRef");
        }
        if (memrefType && memrefType.getElementType().isa<FloatType>()) {
          auto pointeeType =
              LLVM::LLVMPointerType::get(memrefType.getElementType());
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
                                      user->getLoc(), pointeeType, casted,
                                      rewriter.getDenseI64ArrayAttr(0))
                                  .getResult());

          if (constSet.contains(arg_index)) {
            // The aligned pointer must be marked const
            arguments.push_back(enzyme_const_addr.getResult());
          }
          arguments.push_back(rewriter
                                  .create<LLVM::ExtractValueOp>(
                                      user->getLoc(), pointeeType, casted,
                                      rewriter.getDenseI64ArrayAttr(1))
                                  .getResult());

          if (!constSet.contains(arg_index)) {
            // Shadow pointer has to follow the aligned pointer
            Value shadow = *(++it);
            assert(shadow && shadow.getType().isa<MemRefType>() &&
                   "Shadow argument must be a Memref");
            auto shadowCasted =
                rewriter
                    .create<UnrealizedConversionCastOp>(
                        shadow.getLoc(),
                        typeConverter.convertType(shadow.getType()), shadow)
                    .getResult(0);
            auto extractShadowOp = rewriter.create<LLVM::ExtractValueOp>(
                shadow.getLoc(), pointeeType, shadowCasted,
                rewriter.getDenseI64ArrayAttr(1));
            arguments.push_back(extractShadowOp.getResult());
          }

          auto llvmI64Ty = IntegerType::get(user->getContext(), 64);
          arguments.push_back(rewriter
                                  .create<LLVM::ExtractValueOp>(
                                      user->getLoc(), llvmI64Ty, casted,
                                      rewriter.getDenseI64ArrayAttr(2))
                                  .getResult());
          for (int64_t i = 0; i < rank; ++i) {
            arguments.push_back(rewriter
                                    .create<LLVM::ExtractValueOp>(
                                        user->getLoc(), llvmI64Ty, casted,
                                        rewriter.getDenseI64ArrayAttr({3, i}))
                                    .getResult());
          }
          for (int64_t i = 0; i < rank; ++i) {
            arguments.push_back(rewriter
                                    .create<LLVM::ExtractValueOp>(
                                        user->getLoc(), llvmI64Ty, casted,
                                        rewriter.getDenseI64ArrayAttr({4, i}))
                                    .getResult());
          }
        } else {
          arguments.push_back(arg);
        }
        arg_index++;
      }

      rewriter.replaceOpWithNewOp<LLVM::CallOp>(user, primalType.getResults(),
                                                sym, arguments);
      // rewriter.replaceOpWithNewOp<func::CallOp>(
      //     user, sym, primalType.getResults(), arguments);
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
    auto shortTy = IntegerType::get(context, 8);
    rewriter.create<LLVM::GlobalOp>(moduleOp.getLoc(), shortTy,
                                    /*isConstant=*/true,
                                    LLVM::Linkage::Linkonce, "enzyme_const",
                                    IntegerAttr::get(shortTy, 0));
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
    auto voidPtrType = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));

    auto llvmFnType = LLVM::LLVMFunctionType::get(returnType, voidPtrType,
                                                  /*isVarArg=*/true);

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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StandaloneToLLVMLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  StringRef getArgument() const override {
    return "convert-standalone-to-llvm";
  }
  StringRef getDescription() const override {
    return "Lower lagrad.diff calls to Enzyme compatible calls";
  }
  void runOnOperation() final;
};
} // end anonymous namespace

void StandaloneToLLVMLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  target.addIllegalOp<lagrad::DiffOp>();
  target.addLegalDialect<func::FuncDialect>();

  LLVMTypeConverter typeConverter(&getContext());

  RewritePatternSet patterns(&getContext());
  // populateLoopToStdConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  // arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
  patterns.insert<DiffOpLowering>(&getContext());

  auto mod = getOperation();
  if (failed(applyPartialConversion(mod, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::lagrad::createLowerToLLVMPass() {
  return std::make_unique<StandaloneToLLVMLoweringPass>();
}
