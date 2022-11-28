func @insert_overwrite_scalar(%t: tensor<f64>) -> tensor<f64> {
  %cst = arith.constant 2.0 : f64
  %s = tensor.extract %t[] : tensor<f64>
  %m = arith.mulf %cst, %s : f64
  %t_new = tensor.insert %m into %t[] : tensor<f64>
  return %t_new : tensor<f64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %t = arith.constant dense<-1.0> : tensor<f64>
  %f = constant @insert_overwrite_scalar : (tensor<f64>) -> tensor<f64>
  %df = standalone.grad %f : (tensor<f64>) -> tensor<f64>, (tensor<f64>) -> tensor<f64>
  %res = call_indirect %df(%t) : (tensor<f64>) -> tensor<f64>
  %U = tensor.cast %res : tensor<f64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
