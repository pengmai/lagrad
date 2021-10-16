// An autodiff example that includes a function call

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @square(%arg0: f32) -> f32 {
  %0 = arith.mulf %arg0, %arg0 : f32
  return %0 : f32
}

func @mymul(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.mulf %arg0, %arg1 : f32
  return %0 : f32
}

// autograd uses (ans, ...args, g) when constructing VJPs through
// ans is leftover from the forward pass

// function calls. The args are from the forward pass.
func @f(%arg0: f32, %arg1: f32) -> f32 {
  %0 = call @square(%arg0) : (f32) -> f32
  %1 = call @square(%0) : (f32) -> f32
  %2 = call @mymul(%1, %arg1) : (f32, f32) -> f32
  return %2 : f32
}

func @main() {
  %f = constant @f : (f32, f32) -> f32
  %df = standalone.grad %f {of = [0]} : (f32, f32) -> f32, (f32, f32) -> f32

  %cst0 = arith.constant 4.4 : f32
  %cst1 = arith.constant 3.5 : f32
  %res = call_indirect %df(%cst0, %cst1) : (f32, f32) -> f32

  %m = memref.alloca() : memref<f32>
  memref.store %res, %m[] : memref<f32>
  %U = memref.cast %m : memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}
