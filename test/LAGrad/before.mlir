// A very simple example of a program to autodiff.
func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @print_0d(%arg : memref<f32>) {
  %U = memref.cast %arg :  memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

func @square(%arg : f32) -> f32 {
  %res = arith.mulf %arg, %arg : f32
  return %res : f32
}

func @main() -> i64 {
  %cst = arith.constant 3.0 : f32

  %fa = constant @square : (f32) -> f32
  %df = lagrad.diff %fa : (f32) -> f32, (f32) -> f32
  %res = call_indirect %df(%cst) : (f32) -> f32

  %loc = memref.alloca() : memref<f32>
  memref.store %res, %loc[] : memref<f32>
  call @print_0d(%loc) : (memref<f32>) -> ()

  %exit = arith.constant 0 : i64
  return %exit : i64
}