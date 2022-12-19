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
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

def LAGrad_GradOp : LAGrad_Op<"grad", [
        NoSideEffect,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
    let summary = "Reverse-mode autodiff a standard function.";
    let description = [{
        The `lagrad.grad` operation performs a source transformation to compute the gradient of a func op.
    }];

    let arguments = (ins FlatSymbolRefAttr:$F, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>:$results);

    let assemblyFormat = [{
        $F `(` $operands `)` attr-dict `:` functional-type($operands, $results)
    }];
}

def LAGrad_TangentOp : LAGrad_Op<"tangent", [
        NoSideEffect,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
    let summary = "Forward-mode autodiff a standard function.";
    let description = [{
        The `lagrad.tangent` operation performs a source transformation to compute the forward-mode (tangent) gradient of a function.
    }];

    let arguments = (ins FlatSymbolRefAttr:$F, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>:$results);

    let assemblyFormat = [{
        $F `(` $operands `)` attr-dict `:` functional-type($operands, $results)
    }];
}

// TODO: Look into interfaces to automate the verification
def LAGrad_PackOp : LAGrad_Op<"pack", [NoSideEffect]> {
    let summary = "convert fully materialized triangular tensor to packed";
    let description = [{
        The `lagrad.pack` operation denotes a conversion from a fully materialized strictly lower triangular tensor to a packed triangular tensor.
    }];

    let arguments = (ins AnyRankedTensor:$source);
    let results = (outs AnyRankedTensor:$dest);

    let assemblyFormat = [{
        $source attr-dict `:` type($source) `to` type($dest)
    }];

    let verifier = [{ return ::verify(*this); }];
}

#endif // LAGRAD_OPS
