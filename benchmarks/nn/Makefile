CC=$(HOME)/.local/LLVM12/bin/clang-12
OPT=$(HOME)/.local/LLVM12/bin/opt
OPT14=opt
LGOPT=$(HOME)/Research/lagrad/build/bin/lagrad-opt
LLC=$(HOME)/.local/LLVM12/bin/llc

ENZYME_DYLIB=$(HOME)/Research/Enzyme/enzyme/build/Enzyme/LLVMEnzyme-12.dylib
PROFILER_DYLIB=$(HOME)/Research/profile-pass/build/profiler/libProfilerPass.dylib
RUNNER_UTILS=$(HOME)/.local/lib/libmlir_runner_utils.dylib
RPATH=$(HOME)/.local/lib
PREOPT=-O2 -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops

#
# MLIR Passes
#
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
#
# End MLIR Passes
#

_OBJ = main.o read_data.o enzyme_nn.o nn-mlir.o
OBJ = $(patsubst %,tmp/%,$(_OBJ))
export SDKROOT=$(shell xcrun --show-sdk-path)

DEPS = nn.h
tmp/neuralnet.out: $(OBJ) $(DEPS)
	$(CC) $(OBJ) $(RUNNER_UTILS) -O3 -o $@ -rpath $(RPATH)

tmp/%-lldialect.mlir: %.mlir
	$(LGOPT) $< $(PREPROCESS) $(BUFFERIZE) $(LOWER_TO_LLVM) -o $@

tmp/%-mlir.ll: tmp/%-lldialect.mlir
	mlir-translate $< -mlir-to-llvmir -o $@

# tmp/%-mlir-ffast.ll: tmp/%-mlir.ll:
# 	$(OPT14) $< -enable-new-pm=0 -load $(PROFILER_DYLIB) -llfast-math -o $@

tmp/%-mlir.o: tmp/%-mlir.ll
	$(CC) -Wno-override-module -c $^ -O3 -o $@

tmp/%-unopt.ll: %.c $(DEPS)
	$(CC) $< $(PREOPT) -o $@ -S -emit-llvm

tmp/%-raw.ll: tmp/%-unopt.ll
	$(OPT) $^ -load $(ENZYME_DYLIB) -enzyme -o $@ -S

tmp/%-opt.ll: tmp/%-raw.ll
	$(OPT) $^ -O2 -o $@ -S

tmp/enzyme_nn.o: tmp/enzyme_nn-opt.ll
	$(CC) -c $^ -o $@

tmp/%.o: %.c $(DEPS)
	$(CC) $< -c -O3 -ffast-math -o $@

.PHONY: clean
clean:
	rm -f tmp/*
