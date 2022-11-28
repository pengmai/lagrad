// Simple test demonstrating how to pass a custom initial gradient signal.
// Useful for computing Jacobians.

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @custom_grad(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %cst = arith.constant dense<[2.3, -1.1]> : tensor<2xf64>
  %0 = arith.mulf %arg0, %cst : tensor<2xf64>
  return %0 : tensor<2xf64>
}

func @main() {
  %arg = arith.constant dense<0.0> : tensor<2xf64>
  %g = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>

  %res = lagrad.grad @custom_grad(%arg, %g) {grad_signal} : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %U = tensor.cast %res : tensor<2xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
