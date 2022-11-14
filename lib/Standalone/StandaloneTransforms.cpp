#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

// For debugging. This can be removed later.
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace standalone;

struct DiffTransform : public mlir::OpRewritePattern<DiffOp> {
  DiffTransform(mlir::MLIRContext *context)
      : OpRewritePattern<DiffOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(DiffOp op, mlir::PatternRewriter &rewriter) const override {
    llvm::outs() << "Visiting diff op at ";
    op.getLoc().print(llvm::outs());
    llvm::outs() << "\n";

    // auto autodiffRef = rewriter.getSymbolRefAttr("__enzyme_autodiff");
    // autodiffRef.print(llvm::outs());
    // llvm::outs() << "\n";
    // auto newOp = rewriter.create<mlir::LLVM::CallOp>(op->getLoc(),
    // autodiffRef,
    //                                                  rewriter.getF32Type());
    // rewriter.insert(newOp);

    // rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op);
    // op->getResult(0).replaceAllUsesWith(op->getOperand(0));

    return success();
  }
};

struct FooBarSubstitute : public mlir::OpRewritePattern<FooOp> {
  FooBarSubstitute(mlir::MLIRContext *context)
      : OpRewritePattern<FooOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(FooOp op, mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<FooOp>(op, op->getResult(0).getType(),
                                       op->getOperand(0));
    return success();
  }
};

void FooOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<FooBarSubstitute>(context);
  // results.insert<DiffTransform>(context);
}
