#!/bin/bash

# A utility script to quickly jit a file

UTILS="$HOME/.local/lib/libmlir_runner_utils.dylib"
PREPROCESS="-convert-elementwise-to-linalg -canonicalize -linalg-generalize-named-ops -convert-linalg-triangular-to-loops"
# PREPROCESS="-convert-elementwise-to-linalg -canonicalize"
BUFFERIZE="-tensor-constant-bufferize -tensor-bufferize -standalone-bufferize -linalg-bufferize -scf-bufferize -func-bufferize -finalizing-bufferize" # -buffer-deallocation"
LOWER_TO_LOOPS="-convert-linalg-to-affine-loops -lower-affine"
LOWER_TO_LLVM="-convert-scf-to-std -convert-linalg-to-llvm -convert-math-to-llvm -convert-math-to-libm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts -llvm-legalize-for-export"
JIT="mlir-cpu-runner -entry-point-result=void -shared-libs=$UTILS"
# ./build/bin/standalone-opt $1 -take-grads -canonicalize $PREPROCESS $BUFFERIZE $LOWER_TO_LOOPS
./build/bin/standalone-opt "${1:-/dev/stdin}" -take-grads -canonicalize $PREPROCESS $BUFFERIZE $LOWER_TO_LOOPS $LOWER_TO_LLVM | $JIT
# ./build/bin/standalone-opt $1 $PREPROCESS $BUFFERIZE $LOWER_TO_LOOPS $LOWER_TO_LLVM | $JIT
