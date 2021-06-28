// A very simple example of a program to autodiff.
func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @print_0d(%arg : memref<f32>) {
  %U = memref_cast %arg :  memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

func @square(%arg : f32) -> f32 {
  %res = mulf %arg, %arg : f32
  return %res : f32
}

func @main() {
  %cst = constant 3.0 : f32

  %fa = constant @square : (f32) -> f32
  %df = standalone.diff %fa : (f32) -> f32, (f32) -> f32
  %res = call_indirect %df(%cst) : (f32) -> f32

  %loc = alloca() : memref<f32>
  store %res, %loc[] : memref<f32>
  call @print_0d(%loc) : (memref<f32>) -> ()

  return
}
