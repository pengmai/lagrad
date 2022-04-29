func @pow(%x: f64, %i: index) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p_init = arith.constant 1.0 : f64
  %u = memref.alloca() : memref<f64>
  %res = scf.for %iv = %c0 to %i step %c1 iter_args(%p = %p_init) -> f64 {
    memref.store %p, %u[] : memref<f64>
    %U = memref.cast %u : memref<f64> to memref<*xf64>
    call @print_memref_f64(%U) : (memref<*xf64>) -> ()
    %0 = arith.mulf %p, %x : f64
    scf.yield %0 : f64
  }
  return %res : f64
}

func private @print_memref_f64(memref<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %arg = arith.constant 1.3 : f64
  %c4 = arith.constant 4 : index
  %f = constant @pow : (f64, index) -> f64
  %df = standalone.grad %f {of = [0]} : (f64, index) -> f64, (f64, index) -> f64
  %res = call_indirect %df(%arg, %c4) : (f64, index) -> f64
  %u = memref.alloca() : memref<f64>
  memref.store %res, %u[] : memref<f64>
  %U = memref.cast %u : memref<f64> to memref<*xf64>
  call @print_memref_f64(%U) : (memref<*xf64>) -> ()
  return
}
