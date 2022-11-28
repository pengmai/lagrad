// Test the standalone.diff pipeline's ability to differentiate multiple functions.
func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @print_0d(%arg0 : f32) {
  %space = memref.alloca() : memref<f32>
  memref.store %arg0, %space[] : memref<f32>
  %U = memref.cast %space :  memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

func @square(%arg : f32) -> f32 {
  %res = arith.mulf %arg, %arg : f32
  return %res : f32
}

func @cube(%arg : f32) -> f32 {
  %0 = arith.mulf %arg, %arg : f32
  %1 = arith.mulf %0, %arg : f32
  return %1 : f32
}

func @main() -> i64 {
  %cst = arith.constant 3.0 : f32

  %fa = constant @square : (f32) -> f32
  %fb = constant @cube : (f32) -> f32
  %df = standalone.diff %fa : (f32) -> f32, (f32) -> f32
  %dfb = standalone.diff %fb : (f32) -> f32, (f32) -> f32
  %res = call_indirect %df(%cst) : (f32) -> f32
  %resb = call_indirect %dfb(%cst) : (f32) -> f32

  call @print_0d(%res) : (f32) -> ()
  call @print_0d(%resb) : (f32) -> ()

  %exit = arith.constant 0 : i64
  return %exit : i64
}
