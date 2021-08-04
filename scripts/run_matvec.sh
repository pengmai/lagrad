# Will probably want to consolidate these scripts.

SCRIPT_DIR=`echo $(dirname "$0")`
DRIVERS="$SCRIPT_DIR/../test/Driver"
KERNELS="$SCRIPT_DIR/../test/Standalone/kernels"
TMP="$SCRIPT_DIR/../build/tmp"
BIN="$SCRIPT_DIR/../build/bin"

BUFFERIZE="-tensor-constant-bufferize -linalg-bufferize -tensor-bufferize -func-bufferize"
AFFINE_OPTS="-convert-linalg-to-affine-loops -affine-loop-fusion -memref-dataflow-opt -affine-loop-unroll"
# AFFINE_OPTS="-convert-linalg-to-affine-loops"
LOWERING="-convert-linalg-to-affine-loops -finalizing-bufferize -buffer-deallocation -convert-scf-to-std -convert-linalg-to-llvm"
EXPORT="mlir-translate -mlir-to-llvmir"
COMPILE="llc -filetype=obj"

# Compile the Enzyme object file
# ENZYME_DYLIB=/Users/Appleliu/llvm-project/mlir/examples/playground/enzyme_test/LLVMEnzyme-12.dylib
# $BIN/standalone-opt "$KERNELS/enzyme_matvec.mlir" -convert-standalone-to-llvm | $EXPORT > $TMP/ematvec.ll
# opt "$TMP/ematvec.ll" -load "$ENZYME_DYLIB" -enzyme -o "$TMP/postematvec.ll" -S
# llc -filetype=obj "$TMP/postematvec.ll" > "$TMP/ematvec.o"

# For debugging
# $BIN/standalone-opt "$KERNELS/matvec.mlir" -canonicalize -convert-elementwise-to-linalg $BUFFERIZE $AFFINE_OPTS -finalizing-bufferize

$BIN/standalone-opt "$KERNELS/matvec.mlir" -take-grads -canonicalize -convert-elementwise-to-linalg $BUFFERIZE $AFFINE_OPTS $LOWERING | $EXPORT | $COMPILE > $TMP/mmatvec.o

gcc -c $DRIVERS/matvec.c -o $TMP/matvec.o
# gcc $TMP/matvec.o $TMP/mmatvec.o $TMP/ematvec.o -o $TMP/matvec.out
gcc $TMP/matvec.o $TMP/mmatvec.o -o $TMP/matvec.out
$TMP/matvec.out
