//===- LAGradDialect.td - LAGrad dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LAGRAD_DIALECT
#define LAGRAD_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LAGrad dialect definition.
//===----------------------------------------------------------------------===//

def LAGrad_Dialect : Dialect {
    let name = "lagrad";
    let summary = "A dialect for automatic differentiation in tensor MLIR.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::lagrad";
}

//===----------------------------------------------------------------------===//
// Base LAGrad operation definition.
//===----------------------------------------------------------------------===//

class LAGrad_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<LAGrad_Dialect, mnemonic, traits>;

#endif // LAGRAD_DIALECT
