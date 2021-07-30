# Will probably want to consolidate these scripts.

SCRIPT_DIR=`echo $(dirname "$0")`
DRIVERS="$SCRIPT_DIR/../test/Driver"
KERNELS="$SCRIPT_DIR/../test/Standalone/kernels"
TMP="$SCRIPT_DIR/../build/tmp"
BIN="$SCRIPT_DIR/../build/bin"

BUFFERIZE="-tensor-constant-bufferize -linalg-bufferize -tensor-bufferize -func-bufferize"
LOWERING="-convert-linalg-to-loops -finalizing-bufferize -buffer-deallocation -convert-scf-to-std -convert-linalg-to-llvm"
EXPORT="mlir-translate -mlir-to-llvmir"
COMPILE="llc -filetype=obj"

# For debugging
# $BIN/standalone-opt "$KERNELS/matvec.mlir" -take-grads -canonicalize -convert-elementwise-to-linalg $BUFFERIZE $LOWERING

$BIN/standalone-opt "$KERNELS/matvec.mlir" -take-grads -canonicalize -convert-elementwise-to-linalg $BUFFERIZE $LOWERING | $EXPORT | $COMPILE > $TMP/mmatvec.o

gcc -c $DRIVERS/matvec.c -o $TMP/matvec.o
gcc $TMP/matvec.o $TMP/mmatvec.o -o $TMP/matvec.out
$TMP/matvec.out
