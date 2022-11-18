// An AD example that requires differentiating a nested function call.

func.func @square(%arg0: f64) -> f64 {
  %0 = call @mul(%arg0, %arg0) : (f64, f64) -> f64
  return %0 : f64
}

func.func @mul(%arg0: f64, %arg1: f64) -> f64 {
  %0 = arith.mulf %arg0, %arg1 : f64
  return %0 : f64
}

func.func @outer(%arg0: f64) -> f64 {
  %0 = call @square(%arg0) : (f64) -> f64
  return %0 : f64
}

func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }

func.func @main() {
  %f = constant @outer : (f64) -> f64
  %arg = arith.constant 1.2 : f64
  %df = standalone.grad %f : (f64) -> f64, (f64) -> f64

  %res = call_indirect %df(%arg) : (f64) -> f64
  %m = memref.alloca() : memref<f64>
  memref.store %res, %m[] : memref<f64>
  %U = memref.cast %m : memref<f64> to memref<*xf64>
  call @printMemrefF64(%U) : (memref<*xf64>) -> ()
  return
}
