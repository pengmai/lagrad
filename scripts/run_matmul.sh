SCRIPT_DIR=`echo $(dirname "$0")`
DRIVERS="$SCRIPT_DIR/../test/Driver"
KERNELS="$SCRIPT_DIR/../test/Standalone/kernels"
TMP="$SCRIPT_DIR/../build/tmp"
BIN="$SCRIPT_DIR/../build/bin"

BUFFERIZE="-tensor-constant-bufferize -linalg-bufferize -tensor-bufferize -func-bufferize"
AFFINE_OPTS="-convert-linalg-to-affine-loops -affine-loop-unroll"
# AFFINE_OPTS="-convert-linalg-to-affine-loops"
LOWERING="-finalizing-bufferize -buffer-deallocation -convert-scf-to-std -convert-linalg-to-llvm"
EXPORT="mlir-translate -mlir-to-llvmir"
COMPILE="llc -filetype=obj"

# Compile the Enzyme object file
# ENZYME_DYLIB=/Users/Appleliu/llvm-project/mlir/examples/playground/enzyme_test/LLVMEnzyme-12.dylib
# $BIN/standalone-opt "$KERNELS/enzyme_matvec.mlir" -convert-standalone-to-llvm | $EXPORT > $TMP/ematvec.ll
# opt "$TMP/ematvec.ll" -load "$ENZYME_DYLIB" -enzyme -o "$TMP/postematvec.ll" -O3 -S
# llc -filetype=obj "$TMP/postematvec.ll" > "$TMP/ematvec.o"

# For debugging
# $BIN/standalone-opt "$KERNELS/matmul.mlir" #-take-grads -canonicalize $BUFFERIZE -convert-linalg-to-loops

$BIN/standalone-opt "$KERNELS/matmul.mlir" -take-grads -canonicalize $BUFFERIZE $AFFINE_OPTS $LOWERING | $EXPORT | $COMPILE > $TMP/mmatmul.o
# opt -O3 $tmp/mmatvec.ll -o $tmp/mmatvec.ll
# clang -c -O3 $TMP/mmatvec.ll -o $TMP/mmatvec.o
# $COMPILE $TMP/mmatvec.ll -o $TMP/mmatvec.o

gcc -c $DRIVERS/matmul.c -o $TMP/matmul.o
gcc $TMP/matmul.o $TMP/mmatmul.o -o $TMP/matmul.out
$TMP/matmul.out
