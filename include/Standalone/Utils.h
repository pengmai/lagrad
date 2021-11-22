#include "Standalone/StandaloneDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
void runActivityAnalysis(Operation *op, ValueRange args,
                         llvm::SmallDenseSet<Value> &liveset);

Value onesLike(Location loc, Value operand, OpBuilder &builder);

Value constLike(Location loc, Value operand, double scalar, OpBuilder &builder);

Value getZero(Location loc, Value operand, OpBuilder &rewriter);

void collectFreeVars(Block *parentBlock, Region &region,
                     SmallVector<Value> &out);

void eraseUnusedCalls(ModuleOp moduleOp, PatternRewriter &rewriter);

void populateVJP(Operation *op, ModuleOp moduleOp, DenseMap<Value, Value> &env,
                 ConversionPatternRewriter &rewriter);

FuncOp copyFunctionDeclaration(FuncOp funcOp, llvm::StringRef funcName,
                               OpBuilder &rewriter);

FuncOp differentiateFunction(FuncOp funcOp, ArrayAttr gradientsOf,
                             ConversionPatternRewriter &rewriter,
                             bool topLevel);

Value reverseGenericOp(linalg::GenericOp op, Value operand, Value vjp_value,
                       int op_index, ConversionPatternRewriter &rewriter);

Value reverseIfOp(scf::IfOp ifOp, Value freeOperand, Value vjp_value,
                  DenseMap<Value, Value> outer_env,
                  ConversionPatternRewriter &rewriter);

Value reverseForOp(scf::ForOp forOp, Value free_operand, Value vjp_value,
                   size_t result_idx, ConversionPatternRewriter &rewriter);

Value reverseTensorExtractOp(tensor::ExtractOp op, Value operand,
                             Value vjp_value, OpBuilder &builder);

Value reverseCallOp(CallOp op, ModuleOp moduleOp, Value vjp_value,
                    size_t op_index, ConversionPatternRewriter &rewriter);

Value reverseGenericOp(linalg::GenericOp op, Value operand, Value vjp_value,
                       int op_index, ConversionPatternRewriter &rewriter);
} // namespace mlir
