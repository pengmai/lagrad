func @cross(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>, %out: tensor<3xf64>) -> tensor<3xf64> {
  %res = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0) -> ((d0 + 1) mod 3)>,
        affine_map<(d0) -> ((d0 + 2) mod 3)>,
        affine_map<(d0) -> ((d0 + 2) mod 3)>,
        affine_map<(d0) -> ((d0 + 1) mod 3)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%arg0, %arg1, %arg0, %arg1 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>)
    outs(%out : tensor<3xf64>) {
  ^bb0(%arg2: f64, %arg3: f64, %arg4: f64, %arg5: f64, %arg6: f64):
    %0 = arith.mulf %arg2, %arg3 : f64
    %1 = arith.mulf %arg4, %arg5 : f64
    %2 = arith.subf %0, %1 : f64
    linalg.yield %2 : f64
  } -> tensor<3xf64>
  return %res : tensor<3xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %a = arith.constant dense<[1., 2., 3.]> : tensor<3xf64>
  %b = arith.constant dense<[4., 5., 6.]> : tensor<3xf64>
  %da_space = linalg.init_tensor [3] : tensor<3xf64>
  %zero = arith.constant 0.0 : f64
  %da = linalg.fill(%zero, %da_space) : f64, tensor<3xf64> -> tensor<3xf64>
  %out = linalg.init_tensor [3] : tensor<3xf64>
  %f = constant @cross : (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  %df = standalone.grad %f {of = [0], dps = true} : (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>, (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  %res = call_indirect %df(%a, %b, %out, %da) : (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  %U = tensor.cast %res : tensor<3xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
