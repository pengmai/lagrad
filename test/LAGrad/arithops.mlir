// Implement the function -x / (1 + x)
func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

// func @f(%arg0 : f32) -> f32 {
//   %cst = constant 1.0 : f32
//   %0 = addf %cst, %arg0 : f32
//   %1 = divf %arg0, %0 : f32
//   %2 = negf %1 : f32
//   return %2 : f32
// }

func @f(%arg0: f32) -> f32 {
  %0 = mulf %arg0, %arg0 : f32
  return %0 : f32
}

func @main() {
  %cst = constant 3.1 : f32

  %f = constant @f : (f32) -> f32
  %df = standalone.grad %f : (f32) -> f32, (f32) -> f32
  %dval = call_indirect %df(%cst) : (f32) -> f32

  %loc = memref.alloca() : memref<f32>
  memref.store %dval, %loc[] : memref<f32>
  %U = memref.cast %loc : memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

