//
// Based on the implementation from https://github.com/microsoft/ADBench/blob/994fbde50a3ee3c1edc7e7bcdb105470e63d7362/src/python/modules/PyTorch/gmm_objective.py
//

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func @gmm_objective(
  %alphas: tensor<{{k}}xf32>,
  %means: tensor<{{k}}x{{d}}xf32>,
  %Qs: tensor<{{k}}x{{d}}xf32>,
  %Ls: tensor<{{k}}x{{(d * (d-1) / 2) | int}}xf32>,
  %x: tensor<{{n}}x{{d}}xf32>,
  %wishart_gamma: f32,
  %wishart_m: i64
) -> f32 {
  %Qdiags = math.exp %Qs : tensor<{{k}}x{{d}}xf32>

  // Sum along the columns of the Q matrix
  %sum_q_space = constant dense<0.0> : tensor<{{k}}xf32>
  %sum_qs = linalg.generic
    {
      indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%Qdiags: tensor<{{k}}x{{d}}xf32>)
    outs(%sum_q_space: tensor<{{k}}xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<{{k}}xf32>

  %Ls_space = constant dense<0.0> : tensor<{{k}}x{{d}}x{{d}}xf32>
  // %Ls = linalg.generic
  //   {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  //   ins(%)

  // Each datapoint is broadcasted, then the means are elementwise subtracted
  %xcentered_shape = constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf32>
  %xcentered = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    }
    ins(%x, %means : tensor<{{n}}x{{d}}xf32>, tensor<{{k}}x{{d}}xf32>)
    outs(%xcentered_shape : tensor<{{n}}x{{k}}x{{d}}xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = subf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<{{n}}x{{k}}x{{d}}xf32>

  %0 = constant 0.0 : f32
  return %0 : f32
}

func @diag(%x: tensor<{{k}}xf32>) -> tensor<{{k}}x{{k}}xf32> {
  %eye = constant sparse<{{eye}}> : tensor<{{k}}x{{k}}xf32>
  %ones = constant dense<1.0> : tensor<{{k}}xf32>
  %outer_shape = constant dense<0.0> : tensor<{{k}}x{{k}}xf32>
  %outer = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"],
      doc = "vector outer product"
    }
    ins(%x, %ones : tensor<{{k}}xf32>, tensor<{{k}}xf32>)
    outs(%outer_shape: tensor<{{k}}x{{k}}xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = mulf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<{{k}}x{{k}}xf32>
  %out = mulf %eye, %outer : tensor<{{k}}x{{k}}xf32>
  return %out : tensor<{{k}}x{{k}}xf32>
}

func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

#map_all = affine_map<(d0, d1) -> (d0, d1)>
#map_diag = affine_map<(d0, d1) -> (d0)>
func @main() {
  %x = constant dense<{{data}}> : tensor<{{n}}x{{d}}xf32>
  %means = constant dense<{{means}}> : tensor<{{k}}x{{d}}xf32>
  %xcentered_shape = constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf32>
  %out = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    }
    ins(%x, %means : tensor<{{n}}x{{d}}xf32>, tensor<{{k}}x{{d}}xf32>)
    outs(%xcentered_shape : tensor<{{n}}x{{k}}x{{d}}xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = subf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<{{n}}x{{k}}x{{d}}xf32>

  // %out_shape = constant dense<0.0> : tensor<{{k}}x{{k}}xf32>
  // %out = linalg.matmul
  //   ins(%outer, %eye : tensor<{{k}}x{{k}}xf32>, tensor<{{k}}x{{k}}xf32>)
  //   outs(%out_shape : tensor<{{k}}x{{k}}xf32>) -> tensor<{{k}}x{{k}}xf32>

  // %U0 = tensor.cast %lower_tri : tensor<2x2xf32> to tensor<*xf32>
  // call @print_memref_f32(%U0) : (tensor<*xf32>) -> ()
  // %outM = memref.alloc() : memref<2x2xf32>
  // linalg.fill(%cst0, %outM) : f32, memref<2x2xf32>
  // %out_space tensor.load

  // %out = linalg.generic
  //   {indexing_maps = [#map_diag, #map_all], iterator_types = ["parallel", "parallel"]}
  //   ins(%diag : tensor<2xf32>)
  //   outs(%out_space : tensor<2x2xf32>) {
  // ^bb0(%arg0: f32, %arg1: f32):
  //   %0 = addf %arg0, %arg1 : f32
  //   linalg.yield %0 : f32
  // } -> tensor<2x2xf32>
  %U = tensor.cast %out : tensor<{{n}}x{{k}}x{{d}}xf32> to tensor<*xf32>
  call @print_memref_f32(%U) : (tensor<*xf32>) -> ()
  return
}
