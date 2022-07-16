#include "Standalone/StandaloneDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include <string>

namespace mlir {

using ValueSet = llvm::SmallDenseSet<Value>;

class LAGradContext {
public:
  explicit LAGradContext(ModuleOp m) : moduleOp(m) {}
  ModuleOp moduleOp;
  DenseMap<Value, std::string> debug_names;
  ValueSet activeValues;
  ValueSet toBeRecorded;
  DenseMap<Value, Value> tbrCachedVals;
};

bool isFloatOrFloatTensor(Type typ);

void DEBUGpopulateFunc(LAGradContext &ctx, FuncOp funcOp);

void runBottomUpDFS(SmallVector<Value> &frontier, ValueSet &out);
void runTopDownDFS(SmallVector<Value> &frontier, ValueSet &out);

void populatePrimalCaches(LAGradContext &ctx, FuncOp primalFunc,
                          ConversionPatternRewriter &rewriter);

AffineMap getRankReduceSubviewLayout(int64_t resultRank,
                                     ConversionPatternRewriter &rewriter);

void runActivityAnalysis(LAGradContext &ctx, FuncOp primalFunc,
                         ArrayAttr gradientsOf);

SmallVector<Operation *>
cloneBasicBlock(llvm::iterator_range<Region::OpIterator> bbOps,
                OpBuilder &builder, ValueRange regionArgs,
                SmallVector<Value> bbOperands, bool offsetInputs = true,
                LAGradContext *ctx = nullptr);

Value onesLike(Location loc, Value operand, OpBuilder &builder, bool init);

Value constLike(Location loc, Value operand, double scalar, OpBuilder &builder);

Value getZero(Location loc, Value operand, OpBuilder &rewriter,
              bool init = false);

void collectFreeVars(Block *parentBlock, Region &region, ValueSet &out);

void eraseUnusedCalls(ModuleOp moduleOp, PatternRewriter &rewriter);

void populateVJP(Operation *op, LAGradContext &ctx, DenseMap<Value, Value> &env,
                 ConversionPatternRewriter &rewriter);

FuncOp copyFunctionDeclaration(FuncOp funcOp, llvm::StringRef funcName,
                               OpBuilder &rewriter);

FuncOp differentiateFunction(FuncOp funcOp, LAGradContext &ctx,
                             ArrayAttr gradientsOf,
                             ConversionPatternRewriter &rewriter,
                             bool topLevel);

Value reverseGenericOp(linalg::GenericOp op, LAGradContext &ctx, Value operand,
                       Value vjp_value, int op_index,
                       ConversionPatternRewriter &rewriter);

Value reverseIfOp(scf::IfOp ifOp, LAGradContext &ctx, Value freeOperand,
                  Value vjp_value, DenseMap<Value, Value> outer_env,
                  ConversionPatternRewriter &rewriter);

void reverseForOp(scf::ForOp forOp, LAGradContext &ctx, ValueRange free_operand,
                  Value vjp_value, size_t result_idx,
                  DenseMap<Value, Value> &outer_env,
                  ConversionPatternRewriter &rewriter);

Value reverseCallOp(CallOp op, LAGradContext &ctx, Value vjp_value,
                    size_t op_index, ConversionPatternRewriter &rewriter);
} // namespace mlir
