// Implement the function x -> x + x + x

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @add3(%arg : f32) -> f32 {
  %partial = arith.addf %arg, %arg : f32
  %res = arith.addf %arg, %partial : f32
  return %res : f32
}

func @main() {
  %cst = arith.constant 2.1 : f32

  %f = constant @add3 : (f32) -> f32
  %df = standalone.grad %f : (f32) -> f32, (f32) -> f32

  %dval = call_indirect %df(%cst) : (f32) -> f32

  %loc = memref.alloca() : memref<f32>
  memref.store %dval, %loc[] : memref<f32>
  %U = memref.cast %loc : memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}
