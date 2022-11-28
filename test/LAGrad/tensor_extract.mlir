func @extractmul(%arg0: tensor<f64>) -> f64 {
  %cst = arith.constant 3.4 : f64
  %0 = tensor.extract %arg0[] : tensor<f64>
  %1 = arith.mulf %0, %0 : f64
  %2 = arith.mulf %1, %cst : f64
  return %2 : f64
}

func @extract_1d(%arg0: tensor<4xf64>) -> f64 {
  %cst = arith.constant 3.4 : f64
  %idx = arith.constant 3 : index
  %0 = tensor.extract %arg0[%idx] : tensor<4xf64>
  %1 = arith.mulf %0, %0 : f64
  %2 = arith.mulf %1, %cst : f64
  return %2 : f64
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %arg = arith.constant dense<2.3> : tensor<f64>
  %f = constant @extractmul : (tensor<f64>) -> f64
  %df = standalone.grad %f : (tensor<f64>) -> f64, (tensor<f64>) -> tensor<f64>
  %res = call_indirect %df(%arg) : (tensor<f64>) -> tensor<f64>
  %U = tensor.cast %res : tensor<f64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()

  %arg1 = arith.constant dense<[2.3, 3.2, 1.2, 1.1]> : tensor<4xf64>
  %f1 = constant @extract_1d : (tensor<4xf64>) -> f64
  %df1 = standalone.grad %f1 : (tensor<4xf64>) -> f64, (tensor<4xf64>) -> tensor<4xf64>
  %res1 = call_indirect %df1(%arg1) : (tensor<4xf64>) -> tensor<4xf64>
  %U1 = tensor.cast %res1 : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U1) : (tensor<*xf64>) -> ()
  return
}
