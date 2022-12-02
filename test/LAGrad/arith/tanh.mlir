func.func @mytanh(%x: f64) -> f64 {
  %0 = math.tanh %x : f64
  return %0 : f64
}

func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }

func.func @main() {
  %x = arith.constant 3.14159 : f64
  %res = lagrad.grad @mytanh(%x) : (f64) -> f64
  %m = memref.alloca() : memref<f64>
  memref.store %res, %m[] : memref<f64>
  %U = memref.cast %m : memref<f64> to memref<*xf64>
  call @printMemrefF64(%U) : (memref<*xf64>) -> ()
  return
}
