func.func @pow(%x: f64, %n: index) -> f64 {
  %r_init = arith.constant 1.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %res = scf.for %iv = %c0 to %n step %c1 iter_args(%r_outer = %r_init) -> f64 {
    %r_outer_next = scf.for %jv = %c0 to %c3 step %c1 iter_args(%r = %r_outer) -> f64 {
      %r_next = arith.mulf %r, %x : f64
      scf.yield %r_next : f64
    }
    scf.yield %r_outer_next : f64
  }
  return %res : f64
}

// func @__grad_pow(%arg0: f64, %arg1: index) -> f64 {
//   %cst = arith.constant 0.000000e+00 : f64
//   %cst_0 = arith.constant 1.000000e+00 : f64
//   %c2 = arith.constant 2 : index
//   %c1 = arith.constant 1 : index
//   %c0 = arith.constant 0 : index
//   %0 = memref.alloc(%arg1) : memref<?x2xf64>
//   %1 = scf.for %arg2 = %c0 to %arg1 step %c1 iter_args(%arg3 = %cst_0) -> (f64) {
//     %3 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (f64) {
//       memref.store %arg5, %0[%arg2, %arg4] : memref<?x2xf64>
//       %4 = arith.mulf %arg5, %arg0 : f64
//       scf.yield %4 : f64
//     }
//     scf.yield %3 : f64
//   }

//   // So: to handle the nested loop, the cache must be fed through both loop iterations,
//   // and the gradient signal needs to be fed through both nested adjoint loops.
//   %2:2 = scf.for %arg2 = %c0 to %arg1 step %c1 iter_args(%g_outer = %cst_0, %arg3 = %cst) -> (f64, f64) {
//     %3 = arith.subi %arg1, %arg2 : index
//     %4 = arith.subi %3, %c1 : index
//     %8:2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %g_outer, %arg6 = %arg3) -> (f64, f64) {
//       %idx = arith.subi %c1, %arg4 : index

//       %10 = memref.load %0[%4, %idx] : memref<?x2xf64>
//       %11 = arith.mulf %arg5, %arg0 : f64
//       %12 = arith.mulf %arg5, %10 : f64
//       %13 = arith.addf %12, %arg6 : f64
//       scf.yield %11, %13 : f64, f64
//     }
//     scf.yield %8#0, %8#1 : f64, f64
//   }
//   return %2#1 : f64
// }

func.func private @printMemrefF64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func.func @main() {
  %x = arith.constant 1.3 : f64
  %n = arith.constant 4 : index

  %f = constant @pow : (f64, index) -> f64
  %df = standalone.grad %f : (f64, index) -> f64, (f64, index) -> f64
  %res = call_indirect %df(%x, %n) : (f64, index) -> f64
  // %res = call @__grad_pow(%x, %n) : (f64, index) -> f64
  // %res = call @pow(%x, %n) : (f64, index) -> f64
  %t = tensor.empty() : tensor<f64>
  %t_1 = tensor.insert %res into %t[] : tensor<f64>
  %U = tensor.cast %t_1 : tensor<f64> to tensor<*xf64>
  call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  return
}
