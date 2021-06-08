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

def Standalone_FooOp : Standalone_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `standalone.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = standalone.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let hasCanonicalizer = 1;

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def Standalone_DiffOp : Standalone_Op<"diff", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Annotate a function for automatic differentiation";
    let description = [{
        The `standalone.diff` operation annotates a function for automatic differentiation via Enzyme.
    }];

    let arguments = (ins FunctionType:$input);
    let results = (outs FunctionType:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // STANDALONE_OPS
