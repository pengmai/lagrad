SCRIPT_DIR=`echo $(dirname "$0")`
BIN="$SCRIPT_DIR/../build/bin"
MLIR_KERNELS="$SCRIPT_DIR/../test/Standalone/kernels"
DRIVERS="$SCRIPT_DIR/../test/Driver"
TMP_DIR="$SCRIPT_DIR/../build/tmp"
FILE=`echo $1 | cut -d. -f1`
LLVM_FILE="$TMP_DIR/$FILE.ll"
OBJ_FILE="$TMP_DIR/$FILE.o"

# MLIR Flags
TENSOR_PREPROCESSING="-convert-elementwise-to-linalg"
BUFFERIZE="-tensor-constant-bufferize -linalg-bufferize -canonicalize -func-bufferize -tensor-constant-bufferize -tensor-bufferize"
LOWERING="-convert-linalg-to-loops -finalizing-bufferize -convert-linalg-to-llvm"
OPT_ARGS="$TENSOR_PREPROCESSING $BUFFERIZE $LOWERING"

# Compile the Enzyme object file
ENZYME_DYLIB=/Users/Appleliu/llvm-project/mlir/examples/playground/enzyme_test/LLVMEnzyme-12.dylib

clang -S -emit-llvm "$DRIVERS/enzyme_$FILE.c" -o "$TMP_DIR/preenzyme.ll"
opt "$TMP_DIR/preenzyme.ll" -load "$ENZYME_DYLIB" -enzyme -o "$TMP_DIR/postenzyme.ll" -S
clang -c "$TMP_DIR/postenzyme.ll" -o "$TMP_DIR/postenzyme.o"

# For debugging
# $BIN/standalone-opt "$MLIR_KERNELS/$FILE.mlir" $TENSOR_PREPROCESSING $BUFFERIZE

# For execution
$BIN/standalone-opt "$MLIR_KERNELS/$FILE.mlir" -take-grads $OPT_ARGS --llvm-legalize-for-export | mlir-translate -mlir-to-llvmir > "$LLVM_FILE"
llc -filetype=obj < "$LLVM_FILE" > "$OBJ_FILE"

CC=gcc
"$CC" -c "$DRIVERS/$FILE.c" -o "$TMP_DIR/driver.o"
"$CC" "$TMP_DIR/driver.o" "$OBJ_FILE" "$TMP_DIR/postenzyme.o" -o "$TMP_DIR/$FILE.out"

"$TMP_DIR/$FILE.out"
