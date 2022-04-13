// The purpose of this is to test LAGrad's caching of primal values.
func @pow(%x: f64, %cn: i64) -> f64 {
  %r_init = arith.constant 1.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.index_cast %cn : i64 to index
  %r_final = scf.for %iv = %c0 to %n step %c1 iter_args(%r = %r_init) -> (f64) {
    %r_next = arith.mulf %r, %x : f64
    scf.yield %r_next : f64
  }

  return %r_final : f64
}

func private @print_memref_f64(memref<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %arg = arith.constant 1.1 : f64
  %n = arith.constant 9 : i64
  %f = constant @pow : (f64, i64) -> f64
  %df = standalone.grad %f {of = [0]} : (f64, i64) -> f64, (f64, i64) -> f64
  %res = call_indirect %df(%arg, %n) : (f64, i64) -> f64
  %m = memref.alloca() : memref<f64>
  %U = memref.cast %m : memref<f64> to memref<*xf64>
  call @print_memref_f64(%U) : (memref<*xf64>) -> ()
  return
}
