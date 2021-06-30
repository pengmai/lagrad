// Implement the function 4x^2 + x + 4
func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @addandmul(%arg : f32) -> f32 {
  %cst = constant 4.0 : f32
  %0 = mulf %arg, %arg : f32
  %1 = mulf %0, %cst : f32
  %2 = addf %arg, %1 : f32
  %3 = addf %cst, %2 : f32
  return %3 : f32
}

func @main() {
  %cst = constant 0.3 : f32

  %f = constant @addandmul : (f32) -> f32
  %df = standalone.grad %f : (f32) -> f32, (f32) -> f32
  %dval = call_indirect %df(%cst) : (f32) -> f32

  %loc = alloca() : memref<f32>
  store %dval, %loc[] : memref<f32>
  %U = memref_cast %loc : memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}