/**
 * @file LLFastMath.cpp
 * @brief Add fast math flags to supported instructions
 *
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;

namespace {
struct LLFastMathPass : public FunctionPass {
  static char ID;
  LLFastMathPass() : FunctionPass(ID) {}

  virtual bool runOnFunction(Function &F) {
    bool modified = false;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (isa<FPMathOperator>(*I)) {
        I->setFast(true);
        modified = true;
      }
    }
    return modified;
  }
};
} // namespace

char LLFastMathPass::ID = 0;

static RegisterPass<LLFastMathPass> LX("llfast-math", "");

static RegisterStandardPasses LY(PassManagerBuilder::EP_EarlyAsPossible,
                                 [](const PassManagerBuilder &Builder,
                                    legacy::PassManagerBase &PM) {
                                   PM.add(new LLFastMathPass());
                                 });
