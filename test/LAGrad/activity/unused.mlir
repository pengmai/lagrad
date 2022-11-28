func @unused(%arg0: tensor<3xf64>) -> tensor<3xf64> {
  %cst1 = arith.constant dense<-1.2> : tensor<3xf64>
  %cst2 = arith.constant dense<5.4> : tensor<3xf64>
  %1 = arith.mulf %arg0, %cst1 : tensor<3xf64>
  %2 = arith.mulf %arg0, %cst2 : tensor<3xf64>
  return %1 : tensor<3xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes {llvm.emit_c_interface}

func @main() {
  %arg = arith.constant dense<[1., 2., 3.]> : tensor<3xf64>
  %res = lagrad.grad @unused(%arg) : (tensor<3xf64>) -> tensor<3xf64>
  %U = tensor.cast %res : tensor<3xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
