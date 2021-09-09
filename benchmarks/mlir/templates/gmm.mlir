//
// Based on the implementation from https://github.com/microsoft/ADBench/blob/994fbde50a3ee3c1edc7e7bcdb105470e63d7362/src/python/modules/PyTorch/gmm_objective.py
//
func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func @gmm_objective(
  %alphas: tensor<{{k}}xf32>,
  %means: tensor<{{k}}x{{d}}xf32>,
  %Qs: tensor<{{k}}x{{d}}xf32>,
  %Ls: tensor<{{k}}x{{d}}x{{d}}xf32>,
  %x: tensor<{{n}}x{{d}}xf32>,
  %wishart_gamma: f32,
  %wishart_m: i64
) -> tensor<f32> {
  %Qdiags = math.exp %Qs : tensor<{{k}}x{{d}}xf32>

  // Sum along the columns of the Q matrix
  %sum_q_space = constant dense<0.0> : tensor<{{k}}xf32>
  %sum_qs = linalg.generic
    {
      indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%Qs: tensor<{{k}}x{{d}}xf32>)
    outs(%sum_q_space: tensor<{{k}}xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<{{k}}xf32>

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

  // %mysumspace = constant dense<0.0> : tensor<f32>
  // %mysum = linalg.generic
  //   {
  //     indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>],
  //     iterator_types = ["reduction", "reduction"]
  //   }
  //   ins(%x: tensor<{{n}}x{{d}}xf32>)
  //   outs(%mysumspace : tensor<f32>) {
  // ^bb0(%arg0: f32, %arg1: f32):
  //   %0 = addf %arg0, %arg1 : f32
  //   linalg.yield %0 : f32
  // } -> tensor<f32>
  // %mysum = linalg.generic
  //   {
  //     indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>],
  //     iterator_types = ["reduction", "reduction", "reduction"]
  //   }
  //   ins(%xcentered : tensor<{{n}}x{{k}}x{{d}}xf32>)
  //   outs(%mysumspace : tensor<f32>) {
  // ^bb0(%arg0: f32, %arg1: f32):
  //   %0 = addf %arg0, %arg1 : f32
  //   linalg.yield %0 : f32
  // } -> tensor<f32>
  // %U = tensor.cast %mysum : tensor<f32> to tensor<*xf32>
  // call @print_memref_f32(%U) : (tensor<*xf32>) -> ()

  // Compute Qtimesx.
  %Lxcentered_intermediate_shape = constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf32>
  %Lxcentered_intermediate = linalg.generic
    {
      indexing_maps = [
        affine_map<(n, k, d1, d2) -> (k, d1, d2)>,
        affine_map<(n, k, d1, d2) -> (n, k, d2)>,
        affine_map<(n, k, d1, d2) -> (n, k, d1)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    }
    ins(
      %Ls, %xcentered :
      tensor<{{k}}x{{d}}x{{d}}xf32>,
      tensor<{{n}}x{{k}}x{{d}}xf32>
    )
    outs(%Lxcentered_intermediate_shape: tensor<{{n}}x{{k}}x{{d}}xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = mulf %arg0, %arg1 : f32
    %1 = addf %0, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<{{n}}x{{k}}x{{d}}xf32>

  %U = tensor.cast %Lxcentered_intermediate : tensor<{{n}}x{{k}}x{{d}}xf32> to tensor<*xf32>
  call @print_memref_f32(%U) : (tensor<*xf32>) -> ()

  %Lxcentered_shape = constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf32>
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
      tensor<{{k}}x{{d}}xf32>,
      tensor<{{n}}x{{k}}x{{d}}xf32>,
      tensor<{{n}}x{{k}}x{{d}}xf32>
    )
    outs(%Lxcentered_shape : tensor<{{n}}x{{k}}x{{d}}xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
    %0 = mulf %arg0, %arg1 : f32
    %1 = addf %0, %arg2 : f32
    %2 = mulf %1, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<{{n}}x{{k}}x{{d}}xf32>

  %sqsum_Lxcentered_shape = constant dense<0.0> : tensor<{{n}}x{{k}}xf32>
  %sqsum_Lxcentered = linalg.generic
    {indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%Lxcentered : tensor<{{n}}x{{k}}x{{d}}xf32>)
    outs(%sqsum_Lxcentered_shape : tensor<{{n}}x{{k}}xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<{{n}}x{{k}}xf32>

  %alphasPlusSumQs = addf %alphas, %sum_qs : tensor<{{k}}xf32>

  %inner_term_shape = constant dense<0.0> : tensor<{{n}}x{{k}}xf32>
  %inner_term = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%alphasPlusSumQs, %sqsum_Lxcentered : tensor<{{k}}xf32>, tensor<{{n}}x{{k}}xf32>)
    outs(%inner_term_shape : tensor<{{n}}x{{k}}xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %cst = constant 0.5 : f32
    %0 = mulf %arg1, %cst : f32
    %1 = subf %arg0, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<{{n}}x{{k}}xf32>

  %max_init = constant dense<0.0> : tensor<{{n}}xf32>
  %max = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%inner_term : tensor<{{n}}x{{k}}xf32>)
    outs(%max_init : tensor<{{n}}xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = cmpf "ogt", %arg0, %arg1 : f32
    %1 = select %0, %arg0, %arg1 : f32
    linalg.yield %1 : f32
  } -> tensor<{{n}}xf32>

  %lse_init = constant dense<0.0> : tensor<{{n}}xf32>
  %lse_intermediate = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%inner_term, %max : tensor<{{n}}x{{k}}xf32>, tensor<{{n}}xf32>)
    outs(%lse_init : tensor<{{n}}xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = subf %arg0, %arg1 : f32
    %1 = math.exp %0 : f32
    %2 = addf %1, %arg2 : f32
    linalg.yield %2 : f32
  } -> tensor<{{n}}xf32>
  %lse_before_add = math.log %lse_intermediate : tensor<{{n}}xf32>
  %lse = addf %lse_before_add, %max : tensor<{{n}}xf32>
  // %lse = linalg.generic
  //   {
  //     indexing_maps = [],
  //     iterator_types = []
  //   }
  //   ins(%lse_before_sub, %max : tensor<{{n}})
  // %lse_final = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>]}

  %slse_init = constant dense<0.0> : tensor<f32>
  %slse = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    }
    ins(%lse : tensor<{{n}}xf32>)
    outs(%slse_init : tensor<f32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<f32>

  return %slse : tensor<f32>
}

// func @diag(%x: tensor<{{k}}xf32>) -> tensor<{{k}}x{{k}}xf32> {
//   %eye = constant sparse<{{eye}}> : tensor<{{k}}x{{k}}xf32>
//   %ones = constant dense<1.0> : tensor<{{k}}xf32>
//   %outer_shape = constant dense<0.0> : tensor<{{k}}x{{k}}xf32>
//   %outer = linalg.generic
//     {
//       indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
//       iterator_types = ["parallel", "parallel"],
//       doc = "vector outer product"
//     }
//     ins(%x, %ones : tensor<{{k}}xf32>, tensor<{{k}}xf32>)
//     outs(%outer_shape: tensor<{{k}}x{{k}}xf32>) {
//   ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
//     %0 = mulf %arg0, %arg1 : f32
//     linalg.yield %0 : f32
//   } -> tensor<{{k}}x{{k}}xf32>
//   %out = mulf %eye, %outer : tensor<{{k}}x{{k}}xf32>
//   return %out : tensor<{{k}}x{{k}}xf32>
// }

// func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

// func @main() {
//   // %x = constant dense<{{data}}> : tensor<{{n}}x{{d}}xf32>
//   // %means = constant dense<{{means}}> : tensor<{{k}}x{{d}}xf32>
//   %vec = constant dense<[
//     [1., 2.],
//     [3., 4.],
//     [5., 6.]
//   ]> : tensor<3x2xf32>

//   %out_init = constant dense<0.0> : tensor<3xf32>
//   %intermediate = linalg.generic
//     {
//       indexing_maps = [
//         affine_map<(d0, d1) -> (d0, d1)>,
//         affine_map<(d0, d1) -> (d0)>
//       ],
//       iterator_types = ["parallel", "reduction"]
//     }
//     ins(%vec : tensor<3x2xf32>)
//     outs(%out_init: tensor<3xf32>) {
//   ^bb0(%arg0: f32, %arg1: f32):
//     %0 = math.exp %arg0 : f32
//     %1 = addf %0, %arg1 : f32
//     linalg.yield %1 : f32
//   } -> tensor<3xf32>
//   %out = math.log %intermediate : tensor<3xf32>

//   %U = tensor.cast %out : tensor<3xf32> to tensor<*xf32>
//   call @print_memref_f32(%U) : (tensor<*xf32>) -> ()
//   return
// }
