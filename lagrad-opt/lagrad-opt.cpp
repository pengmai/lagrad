//===- lagrad-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "LAGrad/LAGradDialect.h"
#include "LAGrad/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::registerPass(mlir::lagrad::createLowerToLLVMPass);
  mlir::registerPass(mlir::lagrad::createGradPass);
  mlir::registerPass(mlir::lagrad::createElementwiseToAffinePass);
  mlir::registerPass(mlir::lagrad::createBufferizePass);
  mlir::registerPass(mlir::lagrad::createTriangularLoopsPass);
  mlir::registerPass(mlir::lagrad::createPackTriangularPass);
  mlir::registerPass(mlir::lagrad::createStaticAllocsPass);
  mlir::registerPass(mlir::lagrad::createStandaloneDCEPass);
  mlir::registerPass(mlir::lagrad::createLoopHoistingPass);
  mlir::registerPass(mlir::lagrad::createLinalgCanonicalizePass);
  mlir::registerPass(mlir::lagrad::createLinalgToKnownLibraryCallPass);
  mlir::registerPass(mlir::lagrad::createSparsifyPass);

  mlir::DialectRegistry registry;
  registry.insert<mlir::lagrad::LAGradDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated

  registerAllDialects(registry);

  return failed(
      mlir::MlirOptMain(argc, argv, "LAGrad optimizer driver\n", registry));
}
