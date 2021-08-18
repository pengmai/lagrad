# Really need to consolidate these scripts.

SCRIPT_DIR=`echo $(dirname "$0")`
DRIVERS="$SCRIPT_DIR/../test/Driver"
KERNELS="$SCRIPT_DIR/../test/Standalone/kernels"
TMP="$SCRIPT_DIR/../build/tmp"
BIN="$SCRIPT_DIR/../build/bin"

BUFFERIZE="-convert-elementwise-to-linalg -tensor-constant-bufferize -linalg-bufferize -tensor-bufferize -func-bufferize"
AFFINE_OPTS="-convert-linalg-to-affine-loops"
# LOWERING="-finalizing-bufferize -buffer-deallocation -convert-scf-to-std -convert-linalg-to-llvm"
LOWERING="-finalizing-bufferize -convert-scf-to-std -convert-linalg-to-llvm"
EXPORT="mlir-translate -mlir-to-llvmir"
COMPILE="llc -filetype=obj"

# Compile the Enzyme object file
# ENZYME_DYLIB=/Users/Appleliu/llvm-project/mlir/examples/playground/enzyme_test/LLVMEnzyme-12.dylib
# $BIN/standalone-opt "$KERNELS/enzyme_matvec.mlir" -convert-standalone-to-llvm | $EXPORT > $TMP/ematvec.ll
# opt "$TMP/ematvec.ll" -load "$ENZYME_DYLIB" -enzyme -o "$TMP/postematvec.ll" -O3 -S
# llc -filetype=obj "$TMP/postematvec.ll" > "$TMP/ematvec.o"

# For debugging
# $BIN/standalone-opt "$KERNELS/vecmat.mlir" -take-grads -canonicalize $BUFFERIZE -linalg-tile=linalg-tile-sizes=64,1 -convert-linalg-to-loops

$BIN/standalone-opt "$KERNELS/vecmat.mlir" -take-grads -canonicalize $BUFFERIZE $AFFINE_OPTS $LOWERING | $EXPORT | $COMPILE > $TMP/mvecmat.o

gcc -c $DRIVERS/vecmat.c -o $TMP/vecmat.o
gcc $TMP/vecmat.o $TMP/mvecmat.o -o $TMP/vecmat.out
$TMP/vecmat.out
