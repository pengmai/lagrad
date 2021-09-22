//
// Based on the implementation from https://github.com/microsoft/ADBench/blob/994fbde50a3ee3c1edc7e7bcdb105470e63d7362/src/python/modules/PyTorch/gmm_objective.py
//
func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func @gmm_objective(
  %alphas: tensor<{{k}}xf64>,
  %means: tensor<{{k}}x{{d}}xf64>,
  %Qs: tensor<{{k}}x{{d}}xf64>,
  %Ls: tensor<{{k}}x{{d}}x{{d}}xf64>,
  %x: tensor<{{n}}x{{d}}xf64>,
  %wishart_gamma: f64,
  %wishart_m: i64
) -> tensor<f64> {
  %Qdiags = math.exp %Qs : tensor<{{k}}x{{d}}xf64>

  // Sum along the columns of the Q matrix
  %sum_q_space = constant dense<0.0> : tensor<{{k}}xf64>
  %sum_qs = linalg.generic
    {
      indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%Qs: tensor<{{k}}x{{d}}xf64>)
    outs(%sum_q_space: tensor<{{k}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = addf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  } -> tensor<{{k}}xf64>

  // Each datapoint is broadcasted, then the means are elementwise subtracted
  %xcentered_shape = constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf64>
  %xcentered = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    }
    ins(%x, %means : tensor<{{n}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>)
    outs(%xcentered_shape : tensor<{{n}}x{{k}}x{{d}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = subf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  } -> tensor<{{n}}x{{k}}x{{d}}xf64>

  // Compute Qtimesx.
  %Lxcentered_intermediate_shape = constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf64>
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
      tensor<{{k}}x{{d}}x{{d}}xf64>,
      tensor<{{n}}x{{k}}x{{d}}xf64>
    )
    outs(%Lxcentered_intermediate_shape: tensor<{{n}}x{{k}}x{{d}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = mulf %arg0, %arg1 : f64
    %1 = addf %0, %arg2 : f64
    linalg.yield %1 : f64
  } -> tensor<{{n}}x{{k}}x{{d}}xf64>

  // // %subview = tensor.extract_slice %Ls[0, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : tensor<{{k}}x{{d}}x{{d}}xf64> to tensor<1x{{d}}x{{d}}xf64>
  // // %U = tensor.cast %subview : tensor<1x{{d}}x{{d}}xf64> to tensor<*xf64>
  // %subview = tensor.extract_slice %Lxcentered_intermediate[0, 0, 0] [1, 1, {{d}}] [1, 1, 1] : tensor<{{n}}x{{k}}x{{d}}xf64> to tensor<1x1x{{d}}xf64>
  // %U = tensor.cast %subview : tensor<1x1x{{d}}xf64> to tensor<*xf64>
  // call @print_memref_f64(%U) : (tensor<*xf64>) -> ()

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

  // %subview = tensor.extract_slice %Lxcentered[0, 2, 0] [1, 1, {{d}}] [1, 1, 1] : tensor<{{n}}x{{k}}x{{d}}xf64> to tensor<1x1x{{d}}xf64>
  // %U = tensor.cast %subview : tensor<1x1x{{d}}xf64> to tensor<*xf64>
  // call @print_memref_f64(%U) : (tensor<*xf64>) -> ()

  %sqsum_Lxcentered_shape = constant dense<0.0> : tensor<{{n}}x{{k}}xf64>
  %sqsum_Lxcentered = linalg.generic
    {indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%Lxcentered : tensor<{{n}}x{{k}}x{{d}}xf64>)
    outs(%sqsum_Lxcentered_shape : tensor<{{n}}x{{k}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = addf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  } -> tensor<{{n}}x{{k}}xf64>

  %alphasPlusSumQs = addf %alphas, %sum_qs : tensor<{{k}}xf64>

  %inner_term_shape = constant dense<0.0> : tensor<{{n}}x{{k}}xf64>
  %half = constant 0.5 : f64
  %inner_term = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%alphasPlusSumQs, %sqsum_Lxcentered : tensor<{{k}}xf64>, tensor<{{n}}x{{k}}xf64>)
    outs(%inner_term_shape : tensor<{{n}}x{{k}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = mulf %arg1, %half : f64
    %1 = subf %arg0, %0 : f64
    linalg.yield %1 : f64
  } -> tensor<{{n}}x{{k}}xf64>

  // // %sum_inner_term_init = constant dense<0.0> : tensor<f64>
  // // %sum_inner_term = linalg.generic
  // //   {
  // //     indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>],
  // //     iterator_types = ["reduction", "reduction"]
  // //   }
  // //   ins(%inner_term : tensor<{{n}}x{{k}}xf64>)
  // //   outs(%sum_inner_term_init : tensor<f64>) {
  // // ^bb0(%arg0: f64, %arg1: f64):
  // //   %0 = addf %arg0, %arg1 : f64
  // //   linalg.yield %0 : f64
  // // } -> tensor<f64>
  // // return %sum_inner_term : tensor<f64>

  // %max_init = constant dense<0.0> : tensor<{{n}}xf64>
  // %max = linalg.generic
  //   {
  //     indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
  //     iterator_types = ["parallel", "reduction"]
  //   }
  //   ins(%inner_term : tensor<{{n}}x{{k}}xf64>)
  //   outs(%max_init : tensor<{{n}}xf64>) {
  // ^bb0(%arg0: f64, %arg1: f64):
  //   // %0 = cmpf "ogt", %arg0, %arg1 : f64
  //   // %1 = select %0, %arg0, %arg1 : f64
  //   linalg.yield %arg0 : f64
  // } -> tensor<{{n}}xf64>
  
  // // %subview = tensor.extract_slice %inner_term[0, 0][1, {{k}}][1, 1] : tensor<{{n}}x{{k}}xf64> to tensor<1x{{k}}xf64>
  // // %U = tensor.cast %subview : tensor<1x{{k}}xf64> to tensor<*xf64>
  // // call @print_memref_f64(%U) : (tensor<*xf64>) -> ()

  %lse_init = constant dense<0.0> : tensor<{{n}}xf64>
  %lse_intermediate = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        // affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
    // ins(%inner_term, %max : tensor<{{n}}x{{k}}xf64>, tensor<{{n}}xf64>)
    ins(%inner_term : tensor<{{n}}x{{k}}xf64>)
    outs(%lse_init : tensor<{{n}}xf64>) {
  // ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
  //   %0 = subf %arg0, %arg1 : f64
  //   %1 = math.exp %0 : f64
  //   %2 = addf %1, %arg2 : f64
  ^bb0(%arg0: f64, %arg1: f64):
    %1 = math.exp %arg0 : f64
    %2 = addf %1, %arg1 : f64
    linalg.yield %2 : f64
  } -> tensor<{{n}}xf64>
  // %lse_before_add = math.log %lse_intermediate : tensor<{{n}}xf64>
  // %lse = addf %lse_before_add, %max : tensor<{{n}}xf64>
  %lse = math.log %lse_intermediate : tensor<{{n}}xf64>

  %slse_init = constant dense<0.0> : tensor<f64>
  %slse = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    }
    ins(%lse : tensor<{{n}}xf64>)
    outs(%slse_init : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = addf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  } -> tensor<f64>

  // log_wishart_prior
  %Qdiags_summed_init = constant dense<0.0> : tensor<{{k}}xf64>
  %Qdiags_summed = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%Qdiags : tensor<{{k}}x{{d}}xf64>)
    outs(%Qdiags_summed_init : tensor<{{k}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = mulf %arg0, %arg0 : f64
    %1 = addf %0, %arg1 : f64
    linalg.yield %1 : f64
  } -> tensor<{{k}}xf64>

  // This will have a ton of redundant computation
  %L_sq_summed_init = constant dense<0.0> : tensor<{{k}}xf64>
  %L_sq_summed = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction", "reduction"]
    }
    ins(%Ls : tensor<{{k}}x{{d}}x{{d}}xf64>)
    outs(%L_sq_summed_init : tensor<{{k}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = mulf %arg0, %arg0 : f64
    %1 = addf %0, %arg1 : f64
    linalg.yield %1 : f64
  } -> tensor<{{k}}xf64>
  %Qdiags_Lsq_summed = addf %Qdiags_summed, %L_sq_summed : tensor<{{k}}xf64>

  %wishart_out_init = constant dense<0.0> : tensor<f64>
  %wishart_out_1 = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    }
    ins(%Qdiags_Lsq_summed : tensor<{{k}}xf64>)
    outs(%wishart_out_init : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = mulf %half, %wishart_gamma : f64
    %1 = mulf %0, %wishart_gamma : f64
    %2 = mulf %1, %arg0 : f64
    %3 = addf %2, %arg1 : f64
    linalg.yield %3 : f64
  } -> tensor<f64>

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

func @lagrad_gmm( 
  %alphas: tensor<{{k}}xf64>,
  %means: tensor<{{k}}x{{d}}xf64>,
  %Qs: tensor<{{k}}x{{d}}xf64>,
  %Ls: tensor<{{k}}x{{d}}x{{d}}xf64>,
  %x: tensor<{{n}}x{{d}}xf64>,
  %wishart_gamma: f64,
  %wishart_m: i64
) -> (tensor<{{k}}x{{d}}xf64>) {
  %f = constant @gmm_objective : (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64>,
    tensor<{{n}}x{{d}}xf64>,
    f64,
    i64
  ) -> tensor<f64>
  %df = standalone.grad %f {of = [2]}: (
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
  ) -> tensor<{{k}}x{{d}}xf64>
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
  ) -> tensor<{{k}}x{{d}}xf64>
  return %res : tensor<{{k}}x{{d}}xf64>
}

// func @diag(%x: tensor<{{k}}xf64>) -> tensor<{{k}}x{{k}}xf64> {
//   %eye = constant sparse<{{eye}}> : tensor<{{k}}x{{k}}xf64>
//   %ones = constant dense<1.0> : tensor<{{k}}xf64>
//   %outer_shape = constant dense<0.0> : tensor<{{k}}x{{k}}xf64>
//   %outer = linalg.generic
//     {
//       indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
//       iterator_types = ["parallel", "parallel"],
//       doc = "vector outer product"
//     }
//     ins(%x, %ones : tensor<{{k}}xf64>, tensor<{{k}}xf64>)
//     outs(%outer_shape: tensor<{{k}}x{{k}}xf64>) {
//   ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
//     %0 = mulf %arg0, %arg1 : f64
//     linalg.yield %0 : f64
//   } -> tensor<{{k}}x{{k}}xf64>
//   %out = mulf %eye, %outer : tensor<{{k}}x{{k}}xf64>
//   return %out : tensor<{{k}}x{{k}}xf64>
// }

// func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

// func @main() {
//   // %x = constant dense<{{data}}> : tensor<{{n}}x{{d}}xf64>
//   // %means = constant dense<{{means}}> : tensor<{{k}}x{{d}}xf64>
//   %vec = constant dense<[
//     [1., 2.],
//     [3., 4.],
//     [5., 6.]
//   ]> : tensor<3x2xf64>

//   %out_init = constant dense<0.0> : tensor<3xf64>
//   %intermediate = linalg.generic
//     {
//       indexing_maps = [
//         affine_map<(d0, d1) -> (d0, d1)>,
//         affine_map<(d0, d1) -> (d0)>
//       ],
//       iterator_types = ["parallel", "reduction"]
//     }
//     ins(%vec : tensor<3x2xf64>)
//     outs(%out_init: tensor<3xf64>) {
//   ^bb0(%arg0: f64, %arg1: f64):
//     %0 = math.exp %arg0 : f64
//     %1 = addf %0, %arg1 : f64
//     linalg.yield %1 : f64
//   } -> tensor<3xf64>
//   %out = math.log %intermediate : tensor<3xf64>

//   %U = tensor.cast %out : tensor<3xf64> to tensor<*xf64>
//   call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
//   return
// }
