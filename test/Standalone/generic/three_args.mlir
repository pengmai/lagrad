func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @generic_three_args(
  %Qdiags: tensor<{{k}}x{{d}}xf64>,
  %xcentered: tensor<{{n}}x{{k}}x{{d}}xf64>,
  %Lxcentered_intermediate : tensor<{{n}}x{{k}}x{{d}}xf64>
) -> tensor<{{n}}x{{k}}x{{d}}xf64> {
  %Lxcentered_shape = constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf64>
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
    %0 = mulf %arg0, %arg1 : f64
    %1 = addf %0, %arg2 : f64
    %2 = mulf %1, %1 : f64
    linalg.yield %2 : f64
  } -> tensor<{{n}}x{{k}}x{{d}}xf64>
  return %Lxcentered : tensor<{{n}}x{{k}}x{{d}}xf64>
}

func @main() {
  %Qdiags = constant dense<{{Qdiags}}> : tensor<{{k}}x{{d}}xf64>
  %xcentered = constant dense<{{xcentered}}> : tensor<{{n}}x{{k}}x{{d}}xf64>
  %Lxcentered_intermediate = constant dense<{{Lxcentered_intermediate}}> : tensor<{{n}}x{{k}}x{{d}}xf64>

  %f = constant @generic_three_args : (tensor<{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64>
  %df = standalone.grad %f : (tensor<{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64>, (tensor<{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{k}}x{{d}}xf64>
  %res = call_indirect %df(%Qdiags, %xcentered, %Lxcentered_intermediate) : (tensor<{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{k}}x{{d}}xf64>
  %U = tensor.cast %res : tensor<{{k}}x{{d}}xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
