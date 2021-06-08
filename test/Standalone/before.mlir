// A very simple example of a program to autodiff.
func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

// This specifically takes a float and returns a float.
llvm.func @__enzyme_autodiff(!llvm.ptr<func<f32 (f32)>>, ...) -> f32

func @print_0d(%arg : memref<f32>) {
  %U = memref_cast %arg :  memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

func @mul2(%arg : f32) -> f32 {
  %two = constant 2.0 : f32
  %res = mulf %arg, %two : f32
  return %res : f32
}

func @main() {
  %cst = constant 1.3 : f32

  // BEFORE:
  %f = constant @mul2 : (f32) -> f32
  %df = standalone.diff %f : (f32) -> f32
  %res = call_indirect %df(%cst) : (f32) -> f32

  // AFTER:
  // %f = llvm.mlir.addressof @mul2 : !llvm.ptr<func<f32 (f32)>>
  // %res = llvm.call @__enzyme_autodiff(%func, %cst) : (!llvm.ptr<func<f32 (f32)>>, f32) -> f32

  return
}
