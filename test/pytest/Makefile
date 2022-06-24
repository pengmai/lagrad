OPT=../../build/bin/standalone-opt
CC=clang
MLIR_TO_OBJ=mlir-translate -mlir-to-llvmir | llc -filetype=obj
RUNNER_UTILS=$$HOME/.local/lib/libmlir_runner_utils.dylib
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

OBJECTS=\
tmp/ba.o\
tmp/hand_to_pose_params.o\
tmp/hand_get_posed_relatives.o\
tmp/hand_relatives_to_absolutes.o\
tmp/skinned_vertex_subset.o\
tmp/hand_objective.o\
tmp/lstm_model.o\
tmp/lstm_predict.o

tmp/mlir_bindings.dylib: $(OBJECTS)
	clang -dynamiclib -o tmp/mlir_bindings.dylib $(OBJECTS) $(RUNNER_UTILS)

tmp/ba.o: $(MLIR_SRC)/ba/ba.mlir $(OPT)
	$(OPT) $(MLIR_SRC)/ba/ba.mlir $(PREPROCESS) $(BUFFERIZE) $(LOWER_TO_LLVM) | $(MLIR_TO_OBJ) -o tmp/ba.o

tmp/hand_to_pose_params.o: $(MLIR_SRC)/hand/to_pose_params.mlir $(OPT)
	$(OPT) $(MLIR_SRC)/hand/to_pose_params.mlir $(PREPROCESS) $(BUFFERIZE) $(LOWER_TO_LLVM) | $(MLIR_TO_OBJ) -o tmp/hand_to_pose_params.o

tmp/hand_get_posed_relatives.o: $(MLIR_SRC)/hand/get_posed_relatives.mlir $(OPT)
	$(OPT) $(MLIR_SRC)/hand/get_posed_relatives.mlir $(PREPROCESS) $(BUFFERIZE) $(LOWER_TO_LLVM) | $(MLIR_TO_OBJ) -o tmp/hand_get_posed_relatives.o

tmp/hand_relatives_to_absolutes.o: $(MLIR_SRC)/hand/relatives_to_absolutes.mlir $(OPT)
	$(OPT) $(MLIR_SRC)/hand/relatives_to_absolutes.mlir $(PREPROCESS) $(BUFFERIZE) $(LOWER_TO_LLVM) | $(MLIR_TO_OBJ) -o tmp/hand_relatives_to_absolutes.o

tmp/skinned_vertex_subset.o: $(MLIR_SRC)/hand/skinned_vertex_subset.mlir $(OPT)
	$(OPT) $(MLIR_SRC)/hand/skinned_vertex_subset.mlir $(PREPROCESS) $(BUFFERIZE) $(LOWER_TO_LLVM) | $(MLIR_TO_OBJ) -o tmp/skinned_vertex_subset.o

tmp/hand_objective.o: $(MLIR_SRC)/hand/hand_objective.mlir $(OPT)
	$(OPT) $(MLIR_SRC)/hand/hand_objective.mlir $(PREPROCESS) $(BUFFERIZE) $(LOWER_TO_LLVM) | $(MLIR_TO_OBJ) -o tmp/hand_objective.o

tmp/lstm_model.o: $(MLIR_SRC)/lstm/lstm_model.mlir $(OPT)
	$(OPT) $(MLIR_SRC)/lstm/lstm_model.mlir $(PREPROCESS) $(BUFFERIZE) $(LOWER_TO_LLVM) | $(MLIR_TO_OBJ) -o tmp/lstm_model.o

tmp/lstm_predict.o: $(MLIR_SRC)/lstm/lstm_predict.mlir $(OPT)
	$(OPT) $(MLIR_SRC)/lstm/lstm_predict.mlir $(PREPROCESS) $(BUFFERIZE) $(LOWER_TO_LLVM) | $(MLIR_TO_OBJ) -o tmp/lstm_predict.o

clean:
	rm $(OBJECTS) tmp/mlir_bindings.dylib
