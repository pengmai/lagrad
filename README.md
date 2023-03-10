# LAGrad

An MLIR-based source-to-source automatic differentiation system.

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir
cmake --build .
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

## Note for LLVM installation

Make sure to use the custom build scripts in the LLVM repo. Especially make sure to use the `-DCMAKE_INSTALL_PREFIX` flag.

LAGrad is currently pinned to LLVM/MLIR at commit `cb3aa49ec04d4c16863e363bd5d13396d06fffe5`.

## Usage
There are two core ops in LAGrad, `lagrad.tangent` and `lagrad.grad`.
These respectively correspond to taking the derivative of a `FuncOp` using forward mode or reverse mode automatic differentiation.

```mlir
func @square(%x: f64) -> f64 {
  %res = arith.mulf %x, %x : f64
  return %res : f64
}

func @forward_mode_dsquare(%x: f64, %dx: f64) -> f64 {
  %dres = lagrad.tangent @square(%x, %dx) : (f64, f64) -> f64
  return %dres : f64
}

func @reverse_mode_dsquare(%x: f64) -> f64 {
  %dx = lagrad.grad @square(%x) : (f64) -> f64
  return %dx : f64
}
```

LAGrad specializes in tensor MLIR programs, supporting operations in the `linalg`, `tensor`, and `scf` dialects.
```mlir
func @dot(%x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<f64> {
  %init = arith.constant dense<0.0> : tensor<f64>
  %res = linalg.dot ins(%x, %y : tensor<4xf64>, tensor<4xf64>) outs(%init : tensor<f64>) -> tensor<f64>
  return %res : tensor<f64>
}

func @forward_mode_ddot(%x: tensor<4xf64>, %dx: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<f64> {
  %dres = lagrad.tangent @dot(%x, %dx, %y) {of = [0]} : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<f64>)
  return %dres : tensor<f64>
}

func @reverse_mode_ddot(%x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64> {
  %dx = lagrad.grad @dot(%x, %y) {of = [0]} : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  return %dx : tensor<4xf64>
}
```

You can then differentiate these programs using `lagrad-opt`, which is our version of `mlir-opt`:
```sh
lagrad-opt <input_file> -take-grads -canonicalize
```

## Optional Attributes
### Forward Mode (`lagrad.tangent`)
- `of`: `I64ArrayAttr`. The indices of function arguments to take derivatives with respect to. Every argument in this list of indices should have a corresponding seed value.
- `include_primal`: `UnitAttr`. If included, the generated function will return the primal value in addition to the tangent.

### Reverse Mode (`lagrad.grad`)
- `of`: `I64ArrayAttr`. The indices of function arguments to take derivatives with respect to.
- `grad_signal`: `UnitAttr`. If included, the generated function will take a custom seed value as the gradient with respect to the output as its last argument. Otherwise, the value 1 (a scalar or tensor full of ones) will be inferred and used as the seed value.
  - In the tensor case, the default behaviour computes the **elementwise gradient**, or sum along columns of the Jacobian matrix. A custom seed value should be used if the entire Jacobian matrix is desired.
- `sparse`: `UnitAttr`. If included, `grad_signal` must also be included. This tells LAGrad that the seed value is one-hot sparse, meaning it only contains a single nonzero element at a time. This enables what we refer to as **Adjoint Sparsity**, which improves the performance of computing full Jacobians row-by-row.

## Citing
If using LAGrad in an academic context, please cite the following paper:
```
@inproceedings{10.1145/3578360.3580259,
author = {Peng, Mai Jacob and Dubach, Christophe},
title = {LAGrad: Statically Optimized Differentiable Programming in MLIR},
year = {2023},
isbn = {9798400700880},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3578360.3580259},
doi = {10.1145/3578360.3580259},
booktitle = {Proceedings of the 32nd ACM SIGPLAN International Conference on Compiler Construction},
pages = {228–238},
numpages = {11},
keywords = {automatic differentiation, sparsity, MLIR, differentiable programming, static analysis},
location = {Montr\'{e}al, QC, Canada},
series = {CC 2023}
}
```
