//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
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

#include "Standalone/Passes.h"
#include "Standalone/StandaloneDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::registerPass(mlir::Standalone::createLowerToLLVMPass);
  mlir::registerPass(mlir::Standalone::createGradPass);
  mlir::registerPass(mlir::Standalone::createElementwiseToAffinePass);
  mlir::registerPass(mlir::Standalone::createBufferizePass);
  mlir::registerPass(mlir::Standalone::createTriangularLoopsPass);
  mlir::registerPass(mlir::Standalone::createStaticAllocsPass);

  mlir::DialectRegistry registry;
  registry.insert<mlir::standalone::StandaloneDialect>();
  // registry.insert<mlir::StandardOpsDialect>();
  // registry.insert<mlir::linalg::LinalgDialect>();
  // registry.insert<mlir::LLVM::LLVMDialect>();
  // registry.insert<mlir::scf::SCFDialect>();
  // registry.insert<mlir::tensor::TensorDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated

  registerAllDialects(registry);

  return failed(
      mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
