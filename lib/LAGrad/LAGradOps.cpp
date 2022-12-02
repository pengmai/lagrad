//===- LAGradOps.cpp - LAGrad dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LAGrad/LAGradOps.h"
#include "LAGrad/LAGradDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace lagrad;

LogicalResult GradOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("F");
  if (!fnAttr)
    return emitOpError("requires a 'F' symbol reference attribute");
  func::FuncOp fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  bool customGradSignal = (*this)->hasAttrOfType<UnitAttr>("grad_signal");

  if (fnType.getNumInputs() !=
      (getNumOperands() - static_cast<int>(customGradSignal)))
    return emitOpError("incorrect number of operands for F");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  // Verify the function results based on the "gradients of" attribute
  auto gradientsOfAttr = (*this)->getAttrOfType<ArrayAttr>("of");
  SmallVector<int64_t> gradientsOf;
  if (gradientsOfAttr) {
    for (Attribute attr : gradientsOfAttr) {
      if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
        int64_t argIdx = intAttr.getValue().getSExtValue();
        if (argIdx < 0 || argIdx >= fnType.getNumInputs()) {
          return emitOpError("'of' attr given argument index ")
                 << argIdx << ", but function only has "
                 << fnType.getNumInputs() << " inputs.";
        }
        gradientsOf.push_back(argIdx);
      } else {
        return emitOpError("'of' attr was not an integer attribute: ") << attr;
      }
    }
  } else {
    // By default, take the gradient w.r.t. the first argument.
    gradientsOf.push_back(0);
  }

  if (gradientsOf.size() != getNumResults())
    return emitOpError("incorrect number of results");

  for (unsigned i = 0; i < gradientsOf.size(); ++i) {
    if (getResult(i).getType() != fnType.getInput(gradientsOf[i])) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "   op result types: " << getResult(i).getType();
      diag.attachNote() << "function arg types: "
                        << fnType.getInput(gradientsOf[i]);
      return diag;
    }
  }

  return success();
}

LogicalResult TangentOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("F");
  if (!fnAttr)
    return emitOpError("requires a 'F' symbol reference attribute");
  auto fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  return success();
  // // Verify that the operand and result types match the callee.
  // auto fnType = fn.getType();
  // bool customGradSignal = (*this)->hasAttrOfType<UnitAttr>("grad_signal");

  // if (fnType.getNumInputs() !=
  //     (getNumOperands() - static_cast<int>(customGradSignal)))
  //   return emitOpError("incorrect number of operands for F");

  // for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
  //   if (getOperand(i).getType() != fnType.getInput(i))
  //     return emitOpError("operand type mismatch: expected operand type ")
  //            << fnType.getInput(i) << ", but provided "
  //            << getOperand(i).getType() << " for operand number " << i;

  // // Verify the function results based on the "gradients of" attribute
  // auto gradientsOfAttr = (*this)->getAttrOfType<ArrayAttr>("of");
  // SmallVector<int64_t> gradientsOf;
  // if (gradientsOfAttr) {
  //   for (Attribute attr : gradientsOfAttr) {
  //     if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
  //       int64_t argIdx = intAttr.getValue().getSExtValue();
  //       if (argIdx < 0 || argIdx >= fnType.getNumInputs()) {
  //         return emitOpError("'of' attr given argument index ")
  //                << argIdx << ", but function only has "
  //                << fnType.getNumInputs() << " inputs.";
  //       }
  //       gradientsOf.push_back(argIdx);
  //     } else {
  //       return emitOpError("'of' attr was not an integer attribute: ") << attr;
  //     }
  //   }
  // } else {
  //   // By default, take the gradient w.r.t. the first argument.
  //   gradientsOf.push_back(0);
  // }

  // if (gradientsOf.size() != getNumResults())
  //   return emitOpError("incorrect number of results");

  // for (unsigned i = 0; i < gradientsOf.size(); ++i) {
  //   if (getResult(i).getType() != fnType.getInput(gradientsOf[i])) {
  //     auto diag = emitOpError("result type mismatch at index ") << i;
  //     diag.attachNote() << "   op result types: " << getResult(i).getType();
  //     diag.attachNote() << "function arg types: "
  //                       << fnType.getInput(gradientsOf[i]);
  //     return diag;
  //   }
  // }

  // return success();
}

#define GET_OP_CLASSES
#include "LAGrad/LAGradOps.cpp.inc"
