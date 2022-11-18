func.func private @printMemrefF64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func.func @generic_three_args(
  %Qdiags: tensor<{{k}}x{{d}}xf64>,
  %xcentered: tensor<{{n}}x{{k}}x{{d}}xf64>
) -> tensor<{{n}}x{{k}}x{{d}}xf64> {
  %Lxcentered_shape = arith.constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf64>
  %cst = arith.constant dense<2.3> : tensor<{{k}}x{{d}}x{{d}}xf64>
  %Lxcentered_intermediate_init = arith.constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf64>
  %Lxcentered_intermediate = linalg.generic
    {
      indexing_maps = [
        affine_map<(n, k, d1, d2) -> (k, d1, d2)>,
        affine_map<(n, k, d1, d2) -> (n, k, d2)>,
        affine_map<(n, k, d1, d2) -> (n, k, d1)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    }
    ins(%cst, %xcentered : tensor<{{k}}x{{d}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>)
    outs(%Lxcentered_intermediate_init : tensor<{{n}}x{{k}}x{{d}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    linalg.yield %1 : f64
  } -> tensor<{{n}}x{{k}}x{{d}}xf64>

  %Lxcentered = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    }
    ins(
      %Qdiags, %xcentered, %Lxcentered_intermediate :
      tensor<{{k}}x{{d}}xf64>,
      tensor<{{n}}x{{k}}x{{d}}xf64>,
      tensor<{{n}}x{{k}}x{{d}}xf64>
    )
    outs(%Lxcentered_shape : tensor<{{n}}x{{k}}x{{d}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    linalg.yield %1 : f64
  } -> tensor<{{n}}x{{k}}x{{d}}xf64>
  return %Lxcentered : tensor<{{n}}x{{k}}x{{d}}xf64>
}

func.func @main() {
  %Qdiags = arith.constant dense<{{Qdiags}}> : tensor<{{k}}x{{d}}xf64>
  %xcentered = arith.constant dense<{{xcentered}}> : tensor<{{n}}x{{k}}x{{d}}xf64>

  %f = constant @generic_three_args : (tensor<{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64>
  %df = standalone.grad %f {of = [1]}: (tensor<{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64>, (tensor<{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64>
  %res = call_indirect %df(%Qdiags, %xcentered) : (tensor<{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64>
  // %res = call @__grad_generic_three_args(%Qdiags, %xcentered) : (tensor<{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64>

  // Uncomment to test the forward pass
  // %res = call @generic_three_args(%Qdiags, %xcentered) : (tensor<{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64>
  %U = tensor.cast %res : tensor<{{n}}x{{k}}x{{d}}xf64> to tensor<*xf64>

  call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  return
}
