func @pow(%arg0: f64, %arg1: index) -> f64{
  %init = arith.constant 1.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %res = scf.for %iv = %c0 to %arg1 step %c1 iter_args(%r = %init) -> (f64) {
    %r_next = arith.mulf %r, %arg0 : f64
    scf.yield %r_next : f64
  }
  return %res : f64
}

// Primal: 3.1384
// Expected answer: 34.2374
// x^3 -> 3x^2
// func @mygrad_pow(%arg0: f64, %arg1: index) -> f64 {
//   %space = arith.constant 0.000000e+00 : f64
//   %one = arith.constant 1.000000e+00 : f64
//   %primal_init = arith.constant 1.0 : f64
//   %c1 = arith.constant 1 : index
//   %c0 = arith.constant 0 : index
//   %stack_init = linalg.init_tensor [%arg1] : tensor<?xf64>
//   %res:2 = scf.for %iv = %c0 to %arg1 step %c1 iter_args(%r_it = %one, %stack_it = %stack_init) -> (f64, tensor<?xf64>) {
//     %r_next = arith.mulf %r_it, %arg0 : f64
//     %s_next = tensor.insert %r_next into %stack_it[%iv] : tensor<?xf64>
//     scf.yield %r_next, %s_next : f64, tensor<?xf64>
//   }

//   %0:2 = scf.for %iv = %c0 to %arg1 step %c1 iter_args(%g = %one, %dx = %space) -> (f64, f64) {
//     %idx_0 = arith.subi %arg1, %iv : index
//     %idx = arith.subi %idx_0, %c1 : index
//     %r = tensor.extract %res#1[%idx] : tensor<?xf64>

//     %g_next = arith.mulf %g, %arg0 : f64
//     %rtemp = arith.mulf %r, %g : f64
//     %dx_next = arith.addf %dx, %rtemp : f64
//     scf.yield %g_next, %dx_next : f64, f64
//   }
//   return %0#1 : f64
// }

func private @print_memref_f64(memref<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %arg = arith.constant 1.1 : f64
  %n = arith.constant 12 : index
  %f = constant @pow : (f64, index) -> f64
  %df = standalone.grad %f {of = [0]} : (f64, index) -> f64, (f64, index) -> f64
  %res = call_indirect %df(%arg, %n) : (f64, index) -> f64
  // %res = call @mygrad_pow(%arg, %n) : (f64, index) -> f64
  %m = memref.alloca() : memref<f64>
  memref.store %res, %m[] : memref<f64>
  %U = memref.cast %m : memref<f64> to memref<*xf64>
  call @print_memref_f64(%U) : (memref<*xf64>) -> ()
  return
}
