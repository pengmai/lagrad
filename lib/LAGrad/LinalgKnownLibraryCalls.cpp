#include "LAGrad/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using llvm::errs;

namespace {
char getFloatPrefix(unsigned int bitWidth) {
  switch (bitWidth) {
  case 64:
    return 'd';
  case 32:
    return 's';
  case 16:
    return 'h';
  default:
    llvm_unreachable("Unsupported float bitwidth");
  }
}

static MemRefType convertToFullyDynamicMemRef(MemRefType type) {
  SmallVector<int64_t> shape;
  shape.reserve(type.getRank());
  for (int64_t dim = 0; dim < type.getRank(); dim++)
    shape.push_back(-1);
  // TODO: need to tell if we need a fully dynamic layout here
  return MemRefType::get(shape, type.getElementType());
}

static SmallVector<Type, 4> extractOperandTypes(Operation *op) {
  SmallVector<Type, 4> result;
  result.reserve(op->getNumOperands());
  for (auto type : op->getOperandTypes()) {
    // The underlying descriptor type (e.g. LLVM) does not have layout
    // information. Canonicalizing the type at the level of std when going into
    // a library call avoids needing to introduce DialectCastOp.
    if (auto memrefType = type.dyn_cast<MemRefType>()) {
      result.push_back(convertToFullyDynamicMemRef(memrefType));
    } else
      result.push_back(type);
  }
  return result;
}

static SmallVector<Value, 4>
createTypeCanonicalizedMemRefOperands(OpBuilder &b, Location loc,
                                      ValueRange operands) {
  SmallVector<Value, 4> res;
  res.reserve(operands.size());
  for (auto op : operands) {
    auto memrefType = op.getType().dyn_cast<MemRefType>();
    if (!memrefType) {
      res.push_back(op);
      continue;
    }
    Value cast = b.create<memref::CastOp>(
        loc, convertToFullyDynamicMemRef(memrefType), op);
    res.push_back(cast);
  }
  return res;
}

FlatSymbolRefAttr getOrInsertFuncDecl(linalg::LinalgOp linalgOp,
                                      std::string name,
                                      PatternRewriter &rewriter) {
  auto moduleOp = linalgOp->getParentOfType<ModuleOp>();
  auto *ctx = rewriter.getContext();
  FlatSymbolRefAttr fnNameAttr = SymbolRefAttr::get(ctx, name);
  if (moduleOp.lookupSymbol(fnNameAttr.getAttr())) {
    return fnNameAttr;
  }

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto fnType = rewriter.getFunctionType(extractOperandTypes(linalgOp), {});
  auto funcOp = rewriter.create<func::FuncOp>(moduleOp.getLoc(),
                                              fnNameAttr.getValue(), fnType);
  funcOp->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
  funcOp.setPrivate();
  return fnNameAttr;
}

class ReplaceKnownGeneric : public OpRewritePattern<linalg::GenericOp> {
public:
  ReplaceKnownGeneric(MLIRContext *ctx)
      : OpRewritePattern<linalg::GenericOp>(ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (genericOp.hasTensorSemantics() ||
        !genericOp.getLibraryCall().has_value()) {
      // Meant to run after bufferization
      return failure();
    }
    // TODO: Only supports one size at a time, add support for dynamic shapes
    // if/when appropriate
    auto fnNameAttr = getOrInsertFuncDecl(
        genericOp, genericOp.getLibraryCallName(), rewriter);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        genericOp, fnNameAttr, TypeRange(),
        createTypeCanonicalizedMemRefOperands(rewriter, genericOp.getLoc(),
                                              genericOp.getOperands()));
    return success();
  }
};

static unsigned int checkBitWidths(linalg::LinalgOp op) {
  assert(op.getNumDpsInits() > 0);
  unsigned int bitWidth = op.getDpsInputOperand(0)
                              ->get()
                              .getType()
                              .cast<ShapedType>()
                              .getElementTypeBitWidth();
  for (OpOperand *operand : op.getOpOperandsMatchingBBargs()) {
    Type type = operand->get().getType();
    if (auto memrefType = type.dyn_cast<MemRefType>()) {
      if (memrefType.getElementTypeBitWidth() != bitWidth) {
        assert(false &&
               "Expected all linalg MemRef operands to have the same bitwidth");
      }
    } else if (auto floatType = type.dyn_cast<FloatType>()) {
      if (floatType.getIntOrFloatBitWidth() != bitWidth) {
        assert(false &&
               "Expected all linalg Float operands to have the same bitwidth");
      }
    }
  }
  return bitWidth;
}

class ReplaceMatmul : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp mmOp,
                                PatternRewriter &rewriter) const override {
    if (mmOp.hasTensorSemantics()) {
      return failure();
    }

    unsigned int bitWidth = checkBitWidths(mmOp);
    std::string fnName = "dmatmul";
    fnName[0] = getFloatPrefix(bitWidth);

    FlatSymbolRefAttr fnNameAttr = getOrInsertFuncDecl(mmOp, fnName, rewriter);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        mmOp, fnNameAttr, TypeRange(),
        createTypeCanonicalizedMemRefOperands(rewriter, mmOp.getLoc(),
                                              mmOp.getOperands()));
    return success();
  }
};

class ReplaceMatVec : public OpRewritePattern<linalg::MatvecOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatvecOp mvOp,
                                PatternRewriter &rewriter) const override {
    // mvOp.getResult(0).getType().dyn_cast<MemRefType>().getElementTypeBitWidth
    if (mvOp.hasTensorSemantics()) {
      return failure();
    }

    unsigned int bitWidth = checkBitWidths(mvOp);
    std::string fnName = "dmatvec";
    fnName[0] = getFloatPrefix(bitWidth);

    FlatSymbolRefAttr fnNameAttr = getOrInsertFuncDecl(mvOp, fnName, rewriter);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        mvOp, fnNameAttr, TypeRange(),
        createTypeCanonicalizedMemRefOperands(rewriter, mvOp.getLoc(),
                                              mvOp.getOperands()));
    return success();
  }
};

class ReplaceDot : public OpRewritePattern<linalg::DotOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::DotOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasTensorSemantics()) {
      return failure();
    }
    unsigned int bitWidth = checkBitWidths(op);
    // Could add a special case when the input operands are the same buffer to
    // call cblas_dnrm2() ** 2
    std::string fnName = "ddot";
    fnName[0] = getFloatPrefix(bitWidth);
    FlatSymbolRefAttr fnNameAttr = getOrInsertFuncDecl(op, fnName, rewriter);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fnNameAttr, TypeRange(),
        createTypeCanonicalizedMemRefOperands(rewriter, op.getLoc(),
                                              op.getOperands()));
    return success();
  }
};
} // namespace

namespace {
struct LinalgKnownLibraryCallPass
    : public PassWrapper<LinalgKnownLibraryCallPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgKnownLibraryCallPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  StringRef getArgument() const override { return "convert-linalg-to-library"; }
  StringRef getDescription() const override {
    return "Convert linalg ops with a registered library name to those library "
           "calls.";
  }
  void runOnOperation() final {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReplaceKnownGeneric>(patterns.getContext());
    patterns.add<ReplaceMatmul>(patterns.getContext());
    patterns.add<ReplaceMatVec>(patterns.getContext());
    patterns.add<ReplaceDot>(patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
mlir::lagrad::createLinalgToKnownLibraryCallPass() {
  return std::make_unique<LinalgKnownLibraryCallPass>();
}
