func @proj(%A: tensor<3xf64>) -> tensor<2xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.extract %A[%c0] : tensor<3xf64> // s1
  %1 = tensor.extract %A[%c2] : tensor<3xf64> // s2
  %t0 = linalg.init_tensor [2] : tensor<2xf64>
  %t1 = tensor.insert %0 into %t0[%c0] : tensor<2xf64> // s3
  %t2 = tensor.insert %1 into %t1[%c1] : tensor<2xf64> // s4
  return %t2 : tensor<2xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %arg = arith.constant dense<[1., 2., 3.]> : tensor<3xf64>
  %res = lagrad.grad @proj(%arg) : (tensor<3xf64>) -> tensor<3xf64>
  %U = tensor.cast %res : tensor<3xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
