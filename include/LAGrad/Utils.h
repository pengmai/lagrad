#pragma once
#include "LAGrad/LAGradDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include <string>

namespace mlir {
using ValueSet = llvm::SmallDenseSet<Value>;
using NameMap = DenseMap<Value, std::string>;

class LAGradContext {
public:
  explicit LAGradContext(ModuleOp m) : moduleOp(m) {}
  ModuleOp moduleOp;
  NameMap debug_names;
  DenseMap<Value, SmallVector<OpFoldResult>> dynamic_shapes;
  ValueSet activeValues;
  ValueSet effectivelyUsed;
  ValueSet toBeRecorded;
  DenseMap<Value, Value> tbrCachedVals;
};

void printSet(LAGradContext &ctx, const ValueSet &set, bool pretty = true);

bool isFloatOrFloatTensor(Type typ);

bool isLoopParallel(scf::ForOp forOp);

void DEBUGpopulateFunc(NameMap &debug_names, func::FuncOp funcOp);
void analyzeDynamicShapes(LAGradContext &ctx, func::FuncOp funcOp,
                          OpBuilder &builder);

void runBottomUpDFS(SmallVector<Value> &frontier, ValueSet &out);
void runTopDownDFS(SmallVector<Value> &frontier, ValueSet &out);

void populatePrimalCaches(LAGradContext &ctx, func::FuncOp primalFunc,
                          ConversionPatternRewriter &rewriter);

AffineMap getRankReduceSubviewLayout(int64_t resultRank,
                                     ConversionPatternRewriter &rewriter);

void runActivityAnalysis(LAGradContext &ctx, func::FuncOp primalFunc,
                         ArrayAttr gradientsOf);

SmallVector<Operation *>
cloneBasicBlock(llvm::iterator_range<Region::OpIterator> bbOps,
                OpBuilder &builder, ValueRange regionArgs,
                SmallVector<Value> bbOperands, bool offsetInputs = true,
                LAGradContext *ctx = nullptr);

Value onesLike(LAGradContext &ctx, Location loc, Value operand,
               OpBuilder &builder, bool init);

Value constLike(Location loc, Value operand, double scalar, OpBuilder &builder);

Value getZero(Location loc, Value operand, OpBuilder &rewriter,
              bool init = false);

void collectFreeVars(Block *parentBlock, Region &region, ValueSet &out);

void eraseUnusedCalls(ModuleOp moduleOp, PatternRewriter &rewriter);

void populateVJP(Operation *op, LAGradContext &ctx, DenseMap<Value, Value> &env,
                 ConversionPatternRewriter &rewriter);

func::FuncOp copyFunctionDeclaration(func::FuncOp funcOp, llvm::StringRef funcName,
                               OpBuilder &rewriter);

func::FuncOp differentiateFunction(func::FuncOp funcOp, LAGradContext &ctx,
                             ArrayAttr gradientsOf,
                             ConversionPatternRewriter &rewriter, bool topLevel,
                             bool onehotSparse);

Value reverseGenericOp(linalg::GenericOp op, LAGradContext &ctx, Value operand,
                       Value vjp_value, int op_index, Value output,
                       ConversionPatternRewriter &rewriter);

Value reverseIfOp(scf::IfOp ifOp, LAGradContext &ctx, Value freeOperand,
                  Value vjp_value, DenseMap<Value, Value> outer_env,
                  ConversionPatternRewriter &rewriter);

void reverseForOp(scf::ForOp forOp, LAGradContext &ctx, ValueRange free_operand,
                  Value vjp_value, size_t result_idx,
                  DenseMap<Value, Value> &outer_env,
                  ConversionPatternRewriter &rewriter);

Value reverseCallOp(func::CallOp op, LAGradContext &ctx, Value vjp_value,
                    size_t op_index, ConversionPatternRewriter &rewriter);
} // namespace mlir
