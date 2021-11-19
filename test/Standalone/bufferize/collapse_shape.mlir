func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %t = arith.constant dense<[
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
    [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.]
  ]> : tensor<4x10xf64>
  %t1 = arith.constant dense<[
    [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
    [2., 3., 4., 5., 6., 7., 8., 9., 10., 1.]
  ]> : tensor<2x10xf64>
  %cst = arith.constant dense<1.0> : tensor<10xf64>
  %idx = arith.constant 3 : index
  %res = tensor.extract_slice %t[%idx, 0] [1, 10] [1, 1] : tensor<4x10xf64> to tensor<10xf64>
  %res2 = tensor.extract_slice %t1[1, 0] [1, 10] [1, 1] : tensor<2x10xf64> to tensor<10xf64>
  %2 = arith.mulf %res, %res2 : tensor<10xf64>
  %U = tensor.cast %2 : tensor<10xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}