SCRIPT_DIR=`echo $(dirname "$0")`
DRIVERS="$SCRIPT_DIR/../test/Driver"
TMP_DIR="$SCRIPT_DIR/../build/tmp"

gcc "$DRIVERS/dot_reference.c" -O3 -S -emit-llvm -o "$TMP_DIR/dot_reference.ll"
# gcc "$DRIVERS/dot_reference.c" -S -o "$TMP_DIR/dot_reference.s"
# opt "$TMP_DIR/dot_reference.ll" -O3 -o "$TMP_DIR/dot_reference.ll" -S
llc -filetype=obj < "$TMP_DIR/dot_reference.ll" > "$TMP_DIR/dot_reference.o"

gcc "$TMP_DIR/dot_reference.o" -o "$TMP_DIR/dot_reference.out"
# gcc "$DRIVERS/dot_reference.c" -o "$TMP_DIR/dot_reference.out"

"$TMP_DIR/dot_reference.out"
