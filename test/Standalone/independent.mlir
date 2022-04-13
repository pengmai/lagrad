func @independent(%arg0: tensor<4xf64>) -> f64 {
  %zero = arith.constant 0.0 : f64
  return %zero : f64
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %A = arith.constant dense<[1., 2., 3., 4.]> : tensor<4xf64>
  %f = constant @independent : (tensor<4xf64>) -> f64
  %df = standalone.grad %f {of = [0]} : (tensor<4xf64>) -> f64, (tensor<4xf64>) -> tensor<4xf64>
  %res = call_indirect %df(%A) : (tensor<4xf64>) -> tensor<4xf64>
  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
