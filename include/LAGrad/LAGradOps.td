//===- LAGradOps.td - LAGrad dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LAGRAD_OPS
#define LAGRAD_OPS

include "LAGradDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LAGrad_FooOp : LAGrad_Op<"foo", [NoSideEffect,
                                     SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `lagrad.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = lagrad.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let hasCanonicalizer = 1;

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def LAGrad_DiffOp : LAGrad_Op<"diff", [NoSideEffect]> {
    let summary = "Annotate a function for automatic differentiation via Enzyme";
    let description = [{
        The `lagrad.diff` operation annotates a function for automatic differentiation via Enzyme.
    }];

    let arguments = (ins FunctionType:$input);
    let results = (outs FunctionType:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input) `,` type($res)
    }];
}

def LAGrad_GradOp : LAGrad_Op<"grad", [NoSideEffect]> {
    let summary = "Reverse-mode autodiff a standard function.";
    let description = [{
        The `lagrad.grad` operation performs a source transformation to compute the gradient of a std.func.
    }];

    let arguments = (ins FunctionType:$input);
    let results = (outs FunctionType:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input) `,` type($res)
    }];
}

def LAGrad_TangentOp : LAGrad_Op<"tangent", [NoSideEffect]> {
    let summary = "Forward-mode autodiff a standard function.";
    let description = [{
        The `lagrad.tangent` operation performs a source transformation to compute the forward-mode (tangent) gradient of a function.
    }];

    let arguments = (ins FlatSymbolRefAttr:$input);
    let results = (outs FunctionType:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($res)
    }];
}

// TODO: Look into interfaces to automate the verification
def LAGrad_PackOp : LAGrad_Op<"pack", [NoSideEffect]> {
    let summary = "convert fully materialized triangular tensor to packed";
    let description = [{
        The `lagrad.pack` operation denotes a conversion from a fully materialized strictly lower triangular tensor to a packed triangular tensor.
    }];

    let arguments = (ins AnyTensor:$source);
    let results = (outs AnyTensor:$dest);

    let assemblyFormat = [{
        $source attr-dict `:` type($source) `to` type($dest)
    }];
}

#endif // LAGRAD_OPS
