// Copied from gmm_tensor_compressed.mlir, then modified to have full L materialization.
// This is meant to be fully compatible with LAGrad.

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map9 = affine_map<(d0) -> (d0)>
#map10 = affine_map<(d0, d1) -> (d1)>
#map11 = affine_map<(d0) -> ()>
#map12 = affine_map<(d0, d1, d2) -> (d0)>
#map13 = affine_map<() -> ()>
#map14 = affine_map<(d0, d1) -> ()>
module  {
  // func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

  func @mlir_gmm_opt_full(
    %alphas: tensor<{{k}}xf64>,
    %means: tensor<{{k}}x{{d}}xf64>,
    %Qs: tensor<{{k}}x{{d}}xf64>,
    %Ls: tensor<{{k}}x{{d}}x{{d}}xf64>,
    %x: tensor<{{n}}x{{d}}xf64>,
    %wishart_gamma: f64,
    %wishart_m: i64
  ) -> f64 {
    %zero = arith.constant 0.0 : f64
    %Qdiags_space = linalg.init_tensor [{{k}}, {{d}}] : tensor<{{k}}x{{d}}xf64>
    %sum_qs_space = arith.constant dense<0.0> : tensor<{{k}}xf64>
    %len_d_zero = arith.constant dense<0.0> : tensor<{{d}}xf64>
    %trmv_space = linalg.init_tensor [{{d}}] : tensor<{{d}}xf64>
    %main_term_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>
    %zerod_tensor = arith.constant dense<0.0> : tensor<f64>
    %max_space = linalg.init_tensor [] : tensor<f64>

    // This is the preprocess Qs implementation in the original function.
    %Qdiags = linalg.generic
      {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%Qs : tensor<{{k}}x{{d}}xf64>)
      outs(%Qdiags_space : tensor<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = math.exp %arg7 : f64
      linalg.yield %39 : f64
    } -> tensor<{{k}}x{{d}}xf64>

    %sum_qs = linalg.generic
      {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]}
      ins(%Qs : tensor<{{k}}x{{d}}xf64>)
      outs(%sum_qs_space : tensor<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = arith.addf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    } -> tensor<{{k}}xf64>

    %half = arith.constant 0.5 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cn = arith.constant {{n}} : index
    %ck = arith.constant {{k}} : index
    %cd = arith.constant {{d}} : index
    %slse = scf.for %ix = %c0 to %cn step %c1 iter_args(%slse_iv = %zero) -> f64 {
      %main_term = scf.for %ik = %c0 to %ck step %c1 iter_args(%mt_iter = %main_term_space) -> tensor<{{k}}xf64> {
        // Subtract
        %x_slice = tensor.extract_slice %x[%ix, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
        %means_slice = tensor.extract_slice %means[%ik, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
        %xcentered = linalg.generic
          {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]}
          ins(%x_slice, %means_slice : tensor<{{d}}xf64>, tensor<{{d}}xf64>)
          outs(%len_d_zero : tensor<{{d}}xf64>) {
        ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
          %0 = arith.subf %arg0, %arg1 : f64
          linalg.yield %0 : f64
        } -> tensor<{{d}}xf64>

        %Qdiags_slice = tensor.extract_slice %Qdiags[%ik, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
        %Ltri_slice = tensor.extract_slice %Ls[%ik, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : tensor<{{k}}x{{d}}x{{d}}xf64> to tensor<{{d}}x{{d}}xf64>

        // inlined Qtimesx
        // Elementwise multiplication
        %out_0 = arith.mulf %Qdiags_slice, %xcentered : tensor<{{d}}xf64>

        // The triangular matrix-vector multiplication
        %out_1 = linalg.matvec ins(%Ltri_slice, %xcentered : tensor<{{d}}x{{d}}xf64>, tensor<{{d}}xf64>) outs(%len_d_zero : tensor<{{d}}xf64>) -> tensor<{{d}}xf64>
        %Qxcentered = arith.addf %out_0, %out_1 : tensor<{{d}}xf64>

        %msqnorm_t = linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%Qxcentered : tensor<{{d}}xf64>) outs(%zerod_tensor : tensor<f64>) {
        ^bb0(%arg0: f64, %arg1: f64):
          %0 = arith.mulf %arg0, %arg0 : f64
          %1 = arith.addf %0, %arg1 : f64
          linalg.yield %1 : f64
        } -> tensor<f64>
        %msqnorm = tensor.extract %msqnorm_t[] : tensor<f64>
        %hmsqnorm = arith.mulf %msqnorm, %half : f64
        %a_ik = tensor.extract %alphas[%ik] : tensor<{{k}}xf64>
        %q_ik = tensor.extract %sum_qs[%ik] : tensor<{{k}}xf64>
        %sum_aq = arith.addf %a_ik, %q_ik : f64
        %main_term_ik = arith.subf %sum_aq, %hmsqnorm : f64
        %main_term_next = tensor.insert %main_term_ik into %mt_iter[%ik] : tensor<{{k}}xf64>
        scf.yield %main_term_next : tensor<{{k}}xf64>
      }

      // logsumexp %alphas inlined
      // find the max
      %max_init_val = tensor.extract %main_term[%c0] : tensor<{{k}}xf64>
      %max_init = tensor.insert %max_init_val into %max_space[] : tensor<f64>

      %max_t = linalg.generic
        {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
        ins(%main_term : tensor<{{k}}xf64>)
        outs(%max_init : tensor<f64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %p = arith.cmpf "ogt", %arg0, %arg1 : f64
        %next = scf.if %p -> (f64) {
          scf.yield %arg0 : f64
        } else {
          scf.yield %arg1 : f64
        }
        linalg.yield %next : f64
      } -> tensor<f64>

      %max = tensor.extract %max_t[] : tensor<f64>
      %se_noadd_t = linalg.generic
        {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
        ins(%main_term : tensor<{{k}}xf64>)
        outs(%zerod_tensor : tensor<f64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.subf %arg0, %max : f64
        %1 = math.exp %0 : f64
        %2 = arith.addf %1, %arg1 : f64
        linalg.yield %2 : f64
      } -> tensor<f64>
      %se_noadd = tensor.extract %se_noadd_t[] : tensor<f64>
      %lse_noadd = math.log %se_noadd : f64
      %lse = arith.addf %lse_noadd, %max : f64
      // %slse_iter = call @mlogsumexp(%main_term) : (tensor<{{k}}xf64>) -> f64
      %slse_next = arith.addf %slse_iv, %lse : f64
      scf.yield %slse_next : f64
    }

    // inlined logsumexp alphas
    // find the max
    %amax_init_val = tensor.extract %alphas[%c0] : tensor<{{k}}xf64>
    %amax_init = tensor.insert %amax_init_val into %max_space[] : tensor<f64>

    %amax_t = linalg.generic
      {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
      ins(%alphas : tensor<{{k}}xf64>)
      outs(%amax_init : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %p = arith.cmpf "ogt", %arg0, %arg1 : f64
      %next = scf.if %p -> (f64) {
        scf.yield %arg0 : f64
      } else {
        scf.yield %arg1 : f64
      }
      linalg.yield %next : f64
    } -> tensor<f64>

    %amax = tensor.extract %amax_t[] : tensor<f64>
    %ase_noadd_t_out = arith.constant dense<0.0> : tensor<f64>
    %ase_noadd_t = linalg.generic
      {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
      ins(%alphas : tensor<{{k}}xf64>)
      outs(%ase_noadd_t_out : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.subf %arg0, %amax : f64
      %1 = math.exp %0 : f64
      %2 = arith.addf %1, %arg1 : f64
      linalg.yield %2 : f64
    } -> tensor<f64>
    %ase_noadd = tensor.extract %ase_noadd_t[] : tensor<f64>
    %alse_noadd = math.log %ase_noadd : f64
    %lse_alphas = arith.addf %alse_noadd, %amax : f64
    // %lse_alphas = call @mlogsumexp(%alphas) : (tensor<{{k}}xf64>) -> f64

    %cn_float = arith.constant {{n}}.0 : f64
    %nlse_alphas = arith.mulf %cn_float, %lse_alphas : f64

    // log wishart prior inlined
    %c1_i64 = arith.index_cast %c1 : index to i64
    %cd_i64 = arith.index_cast %cd : index to i64
    %n_0 = arith.addi %wishart_m, %c1_i64 : i64
    %n = arith.addi %n_0, %cd_i64 : i64

    %lwishpri = scf.for %ik = %c0 to %ck step %c1 iter_args(%out_iter = %zero) -> (f64) {
      %Qdiags_slice = tensor.extract_slice %Qdiags[%ik, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      // Inlined msqnorm
      %fro_0 = linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%Qdiags_slice : tensor<{{d}}xf64>) outs(%zerod_tensor : tensor<f64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.mulf %arg0, %arg0 : f64
        %1 = arith.addf %0, %arg1 : f64
        linalg.yield %1 : f64
      } -> tensor<f64>
      %frobenius_0 = tensor.extract %fro_0[] : tensor<f64>

      %Ltri_slice = tensor.extract_slice %Ls[%ik, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : tensor<{{k}}x{{d}}x{{d}}xf64> to tensor<{{d}}x{{d}}xf64>
      // Inlined msqnorm_2d
      %fro_1 = linalg.generic {indexing_maps = [#map0, #map14], iterator_types = ["reduction", "reduction"]} ins(%Ltri_slice : tensor<{{d}}x{{d}}xf64>) outs(%zerod_tensor : tensor<f64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.mulf %arg0, %arg0 : f64
        %1 = arith.addf %0, %arg1 : f64
        linalg.yield %1 : f64
      } -> tensor<f64>
      %frobenius_1 = tensor.extract %fro_1[] : tensor<f64>

      %frobenius = arith.addf %frobenius_0, %frobenius_1 : f64

      %out_0 = arith.mulf %wishart_gamma, %wishart_gamma : f64
      %out_1 = arith.mulf %out_0, %half : f64
      %out_2 = arith.mulf %out_1, %frobenius : f64
      %out_3 = arith.sitofp %wishart_m : i64 to f64
      %out_4 = tensor.extract %sum_qs[%ik] : tensor<{{k}}xf64>
      %out_5 = arith.mulf %out_3, %out_4 : f64
      %out_6 = arith.subf %out_2, %out_5 : f64
      %out_next = arith.addf %out_iter, %out_6 : f64
      scf.yield %out_next : f64
    }

    %final_0 = arith.subf %slse, %nlse_alphas : f64
    %final = arith.addf %final_0, %lwishpri : f64
    return %final : f64
  }

  func @lagrad_gmm_full(
    %arg0: tensor<{{k}}xf64>,
    %arg1: tensor<{{k}}x{{d}}xf64>,
    %arg2: tensor<{{k}}x{{d}}xf64>,
    %arg3: tensor<{{k}}x{{d}}x{{d}}xf64>,
    %arg4: tensor<{{n}}x{{d}}xf64>,
    %arg5: f64,
    %arg6: i64
  ) -> (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64>
  ) {
    %zero = arith.constant 0.0 : f64

    %f = constant @mlir_gmm_opt_full : (tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>, tensor<{{n}}x{{d}}xf64>, f64, i64) -> f64
    %df = standalone.grad %f {of = [0, 1, 2, 3]} : (
      tensor<{{k}}xf64>,
      tensor<{{k}}x{{d}}xf64>,
      tensor<{{k}}x{{d}}xf64>,
      tensor<{{k}}x{{d}}x{{d}}xf64>,
      tensor<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> f64, (
      tensor<{{k}}xf64>,
      tensor<{{k}}x{{d}}xf64>,
      tensor<{{k}}x{{d}}xf64>,
      tensor<{{k}}x{{d}}x{{d}}xf64>,
      tensor<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> (
      tensor<{{k}}xf64>,
      tensor<{{k}}x{{d}}xf64>,
      tensor<{{k}}x{{d}}xf64>,
      tensor<{{k}}x{{d}}x{{d}}xf64>
    )

    %res:4 = call_indirect %df(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (
      tensor<{{k}}xf64>,
      tensor<{{k}}x{{d}}xf64>,
      tensor<{{k}}x{{d}}xf64>,
      tensor<{{k}}x{{d}}x{{d}}xf64>,
      tensor<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> (
      tensor<{{k}}xf64>,
      tensor<{{k}}x{{d}}xf64>,
      tensor<{{k}}x{{d}}xf64>,
      tensor<{{k}}x{{d}}x{{d}}xf64>
    )
    return %res#0, %res#1, %res#2, %res#3 : tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>
  }
}
