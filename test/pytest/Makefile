OPT=../../build/bin/standalone-opt
CC=clang
MLIR_TO_OBJ=mlir-translate -mlir-to-llvmir | llc -filetype=obj

PREPROCESS=\
-take-grads\
-canonicalize\
-inline\
-linalg-canonicalize\
-standalone-dce\
-symbol-dce\
-convert-elementwise-to-linalg\
-convert-linalg-triangular-to-loops\
-canonicalize

BUFFERIZE=\
-tensor-constant-bufferize\
-tensor-bufferize\
-standalone-bufferize\
-linalg-bufferize\
-scf-bufferize\
-func-bufferize\
-finalizing-bufferize\
-buffer-hoisting\
-buffer-loop-hoisting\
-standalone-loop-hoisting\
-promote-buffers-to-stack\
-buffer-deallocation\
-canonicalize

LOWER_TO_LLVM=\
-convert-linalg-to-loops\
-lower-affine\
-convert-scf-to-std\
-convert-memref-to-llvm\
-convert-math-to-llvm\
-convert-math-to-libm\
-convert-std-to-llvm\
-reconcile-unrealized-casts\
-llvm-legalize-for-export

MLIR_SRC=../Standalone

OBJECTS=tmp/lstm_model.o tmp/lstm_predict.o

tmp/mlir_bindings.dylib: $(OBJECTS)
	clang -dynamiclib -o tmp/mlir_bindings.dylib $(OBJECTS)

tmp/lstm_model.o: $(MLIR_SRC)/lstm/lstm_model.mlir
	$(OPT) $(MLIR_SRC)/lstm/lstm_model.mlir $(PREPROCESS) $(BUFFERIZE) $(LOWER_TO_LLVM) | $(MLIR_TO_OBJ) -o tmp/lstm_model.o

tmp/lstm_predict.o: $(MLIR_SRC)/lstm/lstm_predict.mlir
	$(OPT) $(MLIR_SRC)/lstm/lstm_predict.mlir $(PREPROCESS) $(BUFFERIZE) $(LOWER_TO_LLVM) | $(MLIR_TO_OBJ) -o tmp/lstm_predict.o
