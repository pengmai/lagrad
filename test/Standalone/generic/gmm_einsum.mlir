func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @gmm_einsum(%arg0: tensor<3x4x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64>{
  %cst = constant dense<0.0> : tensor<2x3x4xf64>
  %0 = linalg.generic
    {
      indexing_maps = [
        affine_map<(n, k, d1, d2) -> (k, d1, d2)>,
        affine_map<(n, k, d1, d2) -> (n, k, d2)>,
        affine_map<(n, k, d1, d2) -> (n, k, d1)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    }
    ins(%arg0, %arg1 : tensor<3x4x4xf64>, tensor<2x3x4xf64>)
    outs(%cst : tensor<2x3x4xf64>) {
  ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
    %0 = mulf %arg2, %arg3 : f64
    %1 = addf %0, %arg4 : f64
    linalg.yield %1 : f64
  } -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

func @main() {
  %a = constant dense<[
    [[ 0.,  1.,  2.,  3.], [ 4.,  5.,  6.,  7.], [ 8.,  9., 10., 11.], [12., 13., 14., 15.]],
    [[16., 17., 18., 19.], [20., 21., 22., 23.], [24., 25., 26., 27.], [28., 29., 30., 31.]],
    [[32., 33., 34., 35.], [36., 37., 38., 39.], [40., 41., 42., 43.], [44., 45., 46., 47.]]
  ]> : tensor<3x4x4xf64>

  %b = constant dense<[
    [[ 0.,  1.,  2.,  3.], [ 4.,  5.,  6.,  7.], [ 8.,  9., 10., 11.]],
    [[12., 13., 14., 15.], [16., 17., 18., 19.], [20., 21., 22., 23.]]
  ]> : tensor<2x3x4xf64>
  %res = call @gmm_einsum(%a, %b) : (tensor<3x4x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>

  %U = tensor.cast %res : tensor<2x3x4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
