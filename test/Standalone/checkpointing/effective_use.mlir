// This is a test to verify that LAGrad can differentiate between
// primal values that are "effectively used" in the adjoint and
// primal values that are not.

func @euse(%x: f64) -> f64 {
  %init = linalg.init_tensor [] : tensor<f64>
  %unused = linalg.init_tensor [] : tensor<f64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%t = %init, %u = %unused) -> (tensor<f64>, tensor<f64>) {
    %0 = tensor.extract %t[] : tensor<f64>
    %1 = arith.mulf %x, %0 : f64
    %2 = tensor.extract %u[] : tensor<f64>
    %3 = arith.addf %1, %2 : f64
    %4 = tensor.insert %3 into %t[] : tensor<f64>
    scf.yield %4, %u : tensor<f64>, tensor<f64>
  }
  %final = tensor.extract %res#0[] : tensor<f64>
  return %final : f64
}

// func @euse(%x: f64) -> f64 {
//   %init = arith.constant 1.0 : f64
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c4 = arith.constant 4 : index
//   %res:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%t = %init, %u = %init) -> (f64, f64) {
//     %1 = arith.mulf %x, %t : f64
//     %3 = arith.addf %1, %u : f64
//     scf.yield %3, %u : f64, f64
//   }
//   return %res#0 : f64
// }

// for i in range(4):
//   t = x * t + u

// for i in reverse_range(4):
//   dx += dt * t_stored
//   dt = dt * x

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %x = arith.constant 3.3 : f64
  %f = constant @euse : (f64) -> f64
  %df = standalone.grad %f : (f64) -> f64, (f64) -> f64
  %res = call_indirect %df(%x) : (f64) -> f64
  %s = linalg.init_tensor [] : tensor<f64>
  %l = tensor.insert %res into %s[] : tensor<f64>
  %u = tensor.cast %l : tensor<f64> to tensor<*xf64>
  call @print_memref_f64(%u) : (tensor<*xf64>) -> ()
  return
}
