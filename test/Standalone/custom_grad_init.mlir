// Simple test demonstrating how to pass a custom initial gradient signal.
// Useful for computing Jacobians.

func.func private @printMemrefF64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func.func @custom_grad(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %cst = arith.constant dense<[2.3, -1.1]> : tensor<2xf64>
  %0 = arith.mulf %arg0, %cst : tensor<2xf64>
  return %0 : tensor<2xf64>
}

func.func @main() {
  %arg = arith.constant dense<0.0> : tensor<2xf64>
  %g = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>

  %f = constant @custom_grad : (tensor<2xf64>) -> tensor<2xf64>
  %df = standalone.grad %f {grad_signal = true} : (tensor<2xf64>) -> tensor<2xf64>, (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %res = call_indirect %df(%arg, %g) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %U = tensor.cast %res : tensor<2xf64> to tensor<*xf64>
  call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  return
}
