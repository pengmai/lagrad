func @mytanh(%x: f64) -> f64 {
  %0 = math.tanh %x : f64
  return %0 : f64
}

func private @print_memref_f64(memref<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %x = arith.constant 3.14159 : f64
  %f = constant @mytanh : (f64) -> f64
  %df = standalone.grad %f : (f64) -> f64, (f64) -> f64
  %res = call_indirect %df(%x) : (f64) -> f64
  %m = memref.alloca() : memref<f64>
  memref.store %res, %m[] : memref<f64>
  %U = memref.cast %m : memref<f64> to memref<*xf64>
  call @print_memref_f64(%U) : (memref<*xf64>) -> ()
  return
}
