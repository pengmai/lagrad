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
