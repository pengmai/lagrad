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

  %wishart_sum_qs_init = constant dense<0.0> : tensor<f64>
  %wishart_sum_qs = linalg.generic
    {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    }
    ins(%sum_qs : tensor<{{k}}xf64>)
    outs(%wishart_sum_qs_init : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = sitofp %wishart_m : i64 to f64
    %1 = mulf %0, %arg0 : f64
    %2 = addf %1, %arg1 : f64
    linalg.yield %2 : f64
  } -> tensor<f64>

  %wishart_out = subf %wishart_out_1, %wishart_sum_qs : tensor<f64>

  // logsumexp alphas
  %sumexp_alphas_init = constant dense<0.0> : tensor<f64>
  %sumexp_alphas = linalg.generic
    {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    }
    ins(%alphas : tensor<{{k}}xf64>)
    outs(%sumexp_alphas_init : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = math.exp %arg0 : f64
    %1 = addf %0, %arg1 : f64
    linalg.yield %1 : f64
  } -> tensor<f64>
  %logsumexp_alphas = math.log %sumexp_alphas : tensor<f64>

  %n_tensor = constant dense<{{n}}.> : tensor<f64>
  %n_logsumexp_alphas = mulf %n_tensor, %logsumexp_alphas : tensor<f64>

  %final_0 = subf %slse, %n_logsumexp_alphas : tensor<f64>
  %final_1 = addf %final_0, %wishart_out : tensor<f64>
  return %final_1 : tensor<f64>
}

// func @lagrad_gmm( 
//   %alphas: tensor<{{k}}xf64>,
//   %means: tensor<{{k}}x{{d}}xf64>,
//   %Qs: tensor<{{k}}x{{d}}xf64>,
//   %Ls: tensor<{{k}}x{{d}}x{{d}}xf64>,
//   %x: tensor<{{n}}x{{d}}xf64>,
//   %wishart_gamma: f64,
//   %wishart_m: i64
// ) -> (tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>) {
//   %f = constant @gmm_objective : (
//     tensor<{{k}}xf64>,
//     tensor<{{k}}x{{d}}xf64>,
//     tensor<{{k}}x{{d}}xf64>,
//     tensor<{{k}}x{{d}}x{{d}}xf64>,
//     tensor<{{n}}x{{d}}xf64>,
//     f64,
//     i64
//   ) -> tensor<f64>
//   %df = standalone.grad %f {of = [0, 1, 2, 3]} : (
//     tensor<{{k}}xf64>,
//     tensor<{{k}}x{{d}}xf64>,
//     tensor<{{k}}x{{d}}xf64>,
//     tensor<{{k}}x{{d}}x{{d}}xf64>,
//     tensor<{{n}}x{{d}}xf64>,
//     f64,
//     i64
//   ) -> tensor<f64>, (
//     tensor<{{k}}xf64>,
//     tensor<{{k}}x{{d}}xf64>,
//     tensor<{{k}}x{{d}}xf64>,
//     tensor<{{k}}x{{d}}x{{d}}xf64>,
//     tensor<{{n}}x{{d}}xf64>,
//     f64,
//     i64
//   ) -> (tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>)
//   %res:4 = call_indirect %df(
//     %alphas,
//     %means,
//     %Qs,
//     %Ls,
//     %x,
//     %wishart_gamma,
//     %wishart_m
//   ) : (
//     tensor<{{k}}xf64>,
//     tensor<{{k}}x{{d}}xf64>,
//     tensor<{{k}}x{{d}}xf64>,
//     tensor<{{k}}x{{d}}x{{d}}xf64>,
//     tensor<{{n}}x{{d}}xf64>,
//     f64,
//     i64
//   ) -> (tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>)
//   return %res#0, %res#1, %res#2, %res#3 : tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>
// }

func @diff_gmm( 
  %alphas: tensor<{{k}}xf64>,
  %means: tensor<{{k}}x{{d}}xf64>,
  %Qs: tensor<{{k}}x{{d}}xf64>,
  %Ls: tensor<{{k}}x{{d}}x{{d}}xf64>,
  %x: tensor<{{n}}x{{d}}xf64>,
  %wishart_gamma: f64,
  %wishart_m: i64
) -> tensor<{{k}}xf64> {
  %f = constant @gmm_objective : (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64>,
    tensor<{{n}}x{{d}}xf64>,
    f64,
    i64
  ) -> tensor<f64>
  %df = standalone.diff %f : (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64>,
    tensor<{{n}}x{{d}}xf64>,
    f64,
    i64
  ) -> tensor<f64>, (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64>,
    tensor<{{n}}x{{d}}xf64>,
    f64,
    i64
  ) -> tensor<{{k}}xf64>
  %res = call_indirect %df(
    %alphas,
    %means,
    %Qs,
    %Ls,
    %x,
    %wishart_gamma,
    %wishart_m
  ) : (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64>,
    tensor<{{n}}x{{d}}xf64>,
    f64,
    i64
  ) -> tensor<{{k}}xf64>
  return %res : tensor<{{k}}xf64>
}
