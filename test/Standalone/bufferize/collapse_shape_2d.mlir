func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %A = arith.constant dense<[
    [[1., 2.], [3., 4.]],
    [[5., 6.], [7., 8.]],
    [[9., 1.], [2., 3.]]
  ]> : tensor<3x2x2xf64>
  %B = arith.constant dense<[[0.5, 0.5], [-2., -2.]]> : tensor<2x2xf64>
  %Bm = memref.buffer_cast %B : memref<2x2xf64>
  %slice = tensor.extract_slice %A[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<3x2x2xf64> to tensor<2x2xf64>

  %result = arith.mulf %slice, %B : tensor<2x2xf64>
  %U = tensor.cast %result : tensor<2x2xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
