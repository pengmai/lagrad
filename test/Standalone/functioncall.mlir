// An autodiff example that includes a function call

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @square(%arg0: f32) -> f32 {
  %0 = mulf %arg0, %arg0 : f32
  return %0 : f32
}

func @mymul(%arg0: f32, %arg1: f32) -> f32 {
  %0 = mulf %arg0, %arg1 : f32
  return %0 : f32
}

func @f(%arg0: f32, %arg1: f32) -> f32 {
  %0 = call @square(%arg0) : (f32) -> f32
  %1 = call @square(%arg0) : (f32) -> f32
  %2 = call @mymul(%0, %1) : (f32, f32) -> f32
  return %2 : f32
}

func @main() {
  %f = constant @f : (f32, f32) -> f32
  %df = standalone.grad %f {of = [0]} : (f32, f32) -> f32, (f32, f32) -> f32

  %cst0 = constant 4.4 : f32
  %cst1 = constant 3.5 : f32
  %res = call_indirect %df(%cst0, %cst1) : (f32, f32) -> f32

  %m = memref.alloca() : memref<f32>
  memref.store %res, %m[] : memref<f32>
  %U = memref.cast %m : memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}
