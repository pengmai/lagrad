func @extractmul(%arg0: tensor<f64>) -> f64 {
  %cst = constant 3.4 : f64
  %0 = tensor.extract %arg0[] : tensor<f64>
  %1 = mulf %0, %0 : f64
  %2 = mulf %1, %cst : f64
  return %2 : f64
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %arg = constant dense<2.3> : tensor<f64>
  %f = constant @extractmul : (tensor<f64>) -> f64
  %df = standalone.grad %f : (tensor<f64>) -> f64, (tensor<f64>) -> tensor<f64>
  %res = call_indirect %df(%arg) : (tensor<f64>) -> tensor<f64>
  %U = tensor.cast %res : tensor<f64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
