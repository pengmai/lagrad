//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "StandaloneDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Standalone_DiffOp : Standalone_Op<"diff", [NoSideEffect]> {
    let summary = "Annotate a function for automatic differentiation via Enzyme";
    let description = [{
        The `standalone.diff` operation annotates a function for automatic differentiation via Enzyme.
    }];

    let arguments = (ins FunctionType:$input);
    let results = (outs FunctionType:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input) `,` type($res)
    }];
}

def Standalone_GradOp : Standalone_Op<"grad", [NoSideEffect]> {
    let summary = "Reverse-mode autodiff a standard function.";
    let description = [{
        The `standalone.grad` operation performs a source transformation to compute the gradient of a std.func.
    }];

    let arguments = (ins FunctionType:$input);
    let results = (outs FunctionType:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input) `,` type($res)
    }];
}

#endif // STANDALONE_OPS
