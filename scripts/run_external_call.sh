SCRIPT_DIR=`echo $(dirname "$0")`
DRIVERS="$SCRIPT_DIR/../test/Driver"
KERNELS="$SCRIPT_DIR/../test/Standalone/kernels"
TMP="$SCRIPT_DIR/../build/tmp"
BIN="$SCRIPT_DIR/../build/bin"

BUFFERIZE="-tensor-constant-bufferize -linalg-bufferize -tensor-bufferize -func-bufferize -finalizing-bufferize"
AFFINE_OPTS="-convert-linalg-to-affine-loops -affine-loop-unroll"
# AFFINE_OPTS="-convert-linalg-to-affine-loops"
LOWERING="-convert-linalg-to-std -convert-memref-to-llvm -convert-std-to-llvm"
EXPORT="mlir-translate -mlir-to-llvmir"
COMPILE="llc -filetype=obj"

# For debugging
# $BIN/standalone-opt "$KERNELS/matvec.mlir" -take-grads -canonicalize $BUFFERIZE -convert-linalg-to-std -convert-memref-to-llvm -convert-std-to-llvm

$BIN/standalone-opt "$KERNELS/matvec.mlir" -take-grads -canonicalize $BUFFERIZE $LOWERING | $EXPORT | $COMPILE > $TMP/externalcall.o

gcc -c -O3 $KERNELS/vecmat_kernel.c -I$HOME/.local/OpenBLAS/include -o $TMP/vecmat_kernel.o
gcc -c $DRIVERS/matvec.c -o $TMP/matvec.o
gcc $TMP/matvec.o $TMP/vecmat_kernel.o $TMP/externalcall.o $HOME/.local/OpenBLAS/lib/libopenblas.a -o $TMP/externalcall.out
$TMP/externalcall.out
