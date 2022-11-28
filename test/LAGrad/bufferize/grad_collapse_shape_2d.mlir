func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @collapsed(%arg0: tensor<4x4x4xf64>) -> tensor<4x4xf64> {
  %0 = tensor.extract_slice %arg0 [2, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4x4xf64> to tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

func @main() {
  %t = arith.constant dense<1.0> : tensor<4x4x4xf64>
  %res = lagrad.grad @collapsed(%t) : (tensor<4x4x4xf64>) -> tensor<4x4x4xf64>
  %U = tensor.cast %res : tensor<4x4x4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
