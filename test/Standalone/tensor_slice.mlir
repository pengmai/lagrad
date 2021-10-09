func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @sum_with_slice(%arg0: tensor<2x2xf64>) -> tensor<f64> {
  %slice = tensor.extract_slice %arg0[0, 1][2, 1][1, 1] : tensor<2x2xf64> to tensor<2xf64>
  %U = tensor.cast %slice : tensor<2xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()

  %sum_init = constant dense<0.0> : tensor<f64>
  // %sum = linalg.generic
  //   {
  //     indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
  //     iterator_types = ["reduction"]
  //   }
  //   ins(%slice : tensor<2xf64>)
  //   outs(%sum_init : tensor<f64>) {
  // ^bb0(%arg1: f64, %arg2: f64):
  //   %0 = addf %arg1, %arg2 : f64
  //   linalg.yield %0 : f64
  // } -> tensor<f64>
  return %sum_init : tensor<f64>
}

func @main() {
  %A = constant dense<[[1., 2.], [3., 4.]]> : tensor<2x2xf64>
  %res = call @sum_with_slice(%A) : (tensor<2x2xf64>) -> tensor<f64>
  %U = tensor.cast %res : tensor<f64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
