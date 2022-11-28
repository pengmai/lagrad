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
  func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }
  func @gmm_objective_full(
    %alphas: tensor<3xf64>,
    %means: tensor<3x2xf64>,
    %Qs: tensor<3x2xf64>,
    %Ls: tensor<3x2x2xf64>,
    %x: tensor<1x2xf64>,
    %wishart_gamma: f64,
    %wishart_m: i64
  ) -> f64 {
    %zero = arith.constant 0.0 : f64
    %Qdiags_space = linalg.init_tensor [3, 2] : tensor<3x2xf64>
    %sum_qs_space = arith.constant dense<0.0> : tensor<3xf64>
    %len_d_zero = arith.constant dense<0.0> : tensor<2xf64>
    %trmv_space = linalg.init_tensor [2] : tensor<2xf64>
    %main_term_space = linalg.init_tensor [3] : tensor<3xf64>
    %zerod_tensor = arith.constant dense<0.0> : tensor<f64>
    %max_space = linalg.init_tensor [] : tensor<f64>

    // This is the preprocess Qs implementation in the original function.
    %Qdiags = linalg.generic
      {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%Qs : tensor<3x2xf64>)
      outs(%Qdiags_space : tensor<3x2xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = math.exp %arg7 : f64
      linalg.yield %39 : f64
    } -> tensor<3x2xf64>

    %sum_qs = linalg.generic
      {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]}
      ins(%Qs : tensor<3x2xf64>)
      outs(%sum_qs_space : tensor<3xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = arith.addf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    } -> tensor<3xf64>

    %half = arith.constant 0.5 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cn = arith.constant 1 : index
    %ck = arith.constant 3 : index
    %cd = arith.constant 2 : index
    %slse = scf.for %ix = %c0 to %cn step %c1 iter_args(%slse_iv = %zero) -> f64 {
      %main_term = scf.for %ik = %c0 to %ck step %c1 iter_args(%mt_iter = %main_term_space) -> tensor<3xf64> {
        // Subtract
        %x_slice = tensor.extract_slice %x[%ix, 0] [1, 2] [1, 1] : tensor<1x2xf64> to tensor<2xf64>
        %means_slice = tensor.extract_slice %means[%ik, 0] [1, 2] [1, 1] : tensor<3x2xf64> to tensor<2xf64>
        %xcentered = arith.subf %x_slice, %means_slice {lagrad_should_cache} : tensor<2xf64>

        %Qdiags_slice = tensor.extract_slice %Qdiags[%ik, 0] [1, 2] [1, 1] : tensor<3x2xf64> to tensor<2xf64>
        %Ltri_slice = tensor.extract_slice %Ls[%ik, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<3x2x2xf64> to tensor<2x2xf64>

        // inlined Qtimesx
        %p0 = arith.cmpi eq, %ix, %c0 : index
        %p1 = arith.cmpi eq, %ik, %c0 : index
        %p = arith.andi %p0, %p1 : i1
        // Elementwise multiplication
        %out_0 = arith.mulf %Qdiags_slice, %xcentered : tensor<2xf64>

        // The triangular matrix-vector multiplication
        %out_1 = linalg.generic
          {
            doc = "Triangular Matrix-Vector Multiplication",
            indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>],
            iterator_types = ["reduction", "parallel"]
          }
          ins(%Ltri_slice, %xcentered : tensor<2x2xf64>, tensor<2xf64>)
          outs(%len_d_zero : tensor<2xf64>) {
        ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
          %0 = arith.mulf %arg0, %arg1 : f64
          %1 = arith.addf %0, %arg2 : f64
          linalg.yield %1 : f64
        } -> tensor<2xf64>

        %Qxcentered = arith.addf %out_0, %out_1 {lagrad_should_cache} : tensor<2xf64>
        %msqnorm_t = linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%Qxcentered : tensor<2xf64>) outs(%zerod_tensor : tensor<f64>) {
        ^bb0(%arg0: f64, %arg1: f64):
          %0 = arith.mulf %arg0, %arg0 : f64
          %1 = arith.addf %0, %arg1 : f64
          linalg.yield %1 : f64
        } -> tensor<f64>

        %msqnorm = tensor.extract %msqnorm_t[] : tensor<f64>
        %hmsqnorm = arith.mulf %msqnorm, %half : f64
        %a_ik = tensor.extract %alphas[%ik] : tensor<3xf64>
        %q_ik = tensor.extract %sum_qs[%ik] : tensor<3xf64>
        %sum_aq = arith.addf %a_ik, %q_ik : f64
        %main_term_ik = arith.subf %sum_aq, %hmsqnorm : f64
        %main_term_next = tensor.insert %main_term_ik into %mt_iter[%ik] : tensor<3xf64>
        scf.yield %main_term_next : tensor<3xf64>
      } {lagrad_should_cache}

      // logsumexp %alphas inlined
      // find the max
      %max_init_val = tensor.extract %main_term[%c0] : tensor<3xf64>
      %max_init = tensor.insert %max_init_val into %max_space[] : tensor<f64>

      %max_t = linalg.generic
        {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
        ins(%main_term : tensor<3xf64>)
        outs(%max_init : tensor<f64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %p = arith.cmpf "ogt", %arg0, %arg1 : f64
        %next = select %p, %arg0, %arg1 : f64
        linalg.yield %next : f64
      } -> tensor<f64>

      %max = tensor.extract %max_t[] : tensor<f64>
      %se_noadd_t = linalg.generic
        {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
        ins(%main_term : tensor<3xf64>)
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
      %slse_next = arith.addf %slse_iv, %lse : f64
      scf.yield %slse_next : f64
    }

    // inlined logsumexp alphas
    // find the max
    %amax_init_val = tensor.extract %alphas[%c0] : tensor<3xf64>
    %amax_init = tensor.insert %amax_init_val into %max_space[] : tensor<f64>

    %amax_t = linalg.generic
      {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
      ins(%alphas : tensor<3xf64>)
      outs(%amax_init : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %p = arith.cmpf "ogt", %arg0, %arg1 : f64
      %next = select %p, %arg0, %arg1 : f64
      linalg.yield %next : f64
    } -> tensor<f64>

    %amax = tensor.extract %amax_t[] : tensor<f64>
    %ase_noadd_t_out = arith.constant dense<0.0> : tensor<f64>
    %ase_noadd_t = linalg.generic
      {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
      ins(%alphas : tensor<3xf64>)
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

    %cn_float = arith.constant 1.0 : f64
    %nlse_alphas = arith.mulf %cn_float, %lse_alphas : f64

    // log wishart prior inlined
    %c1_i64 = arith.index_cast %c1 : index to i64
    %cd_i64 = arith.index_cast %cd : index to i64
    %n_0 = arith.addi %wishart_m, %c1_i64 : i64
    %n = arith.addi %n_0, %cd_i64 : i64

    %lwishpri = scf.for %ik = %c0 to %ck step %c1 iter_args(%out_iter = %zero) -> (f64) {
      %Qdiags_slice = tensor.extract_slice %Qdiags[%ik, 0] [1, 2] [1, 1] : tensor<3x2xf64> to tensor<2xf64>
      // Inlined msqnorm
      %Qsquared = arith.mulf %Qdiags_slice, %Qdiags_slice {lagrad_should_cache} : tensor<2xf64>
      %fro_0 = linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%Qsquared : tensor<2xf64>) outs(%zerod_tensor : tensor<f64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %1 = arith.addf %arg0, %arg1 : f64
        linalg.yield %1 : f64
      } -> tensor<f64>
      %frobenius_0 = tensor.extract %fro_0[] : tensor<f64>

      %Ltri_slice = tensor.extract_slice %Ls[%ik, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<3x2x2xf64> to tensor<2x2xf64>
      // Inlined msqnorm_2d
      %fro_1 = linalg.generic {indexing_maps = [#map0, #map14], iterator_types = ["reduction", "reduction"]} ins(%Ltri_slice : tensor<2x2xf64>) outs(%zerod_tensor : tensor<f64>) {
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
      %out_4 = tensor.extract %sum_qs[%ik] : tensor<3xf64>
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
    %arg0: tensor<3xf64>,
    %arg1: tensor<3x2xf64>,
    %arg2: tensor<3x2xf64>,
    %arg3: tensor<3x2x2xf64>,
    %arg4: tensor<1x2xf64>,
    %arg5: f64,
    %arg6: i64
  ) -> (
    tensor<3xf64>,
    tensor<3x2xf64>,
    tensor<3x2xf64>,
    tensor<3x2x2xf64>
  ) {
    %zero = arith.constant 0.0 : f64

    %res:4 = lagrad.grad @gmm_objective_full(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) {of = [0, 1, 2, 3]} : (
      tensor<3xf64>,
      tensor<3x2xf64>,
      tensor<3x2xf64>,
      tensor<3x2x2xf64>,
      tensor<1x2xf64>,
      f64,
      i64
    ) -> (
      tensor<3xf64>,
      tensor<3x2xf64>,
      tensor<3x2xf64>,
      tensor<3x2x2xf64>
    )
    return %res#0, %res#1, %res#2, %res#3 : tensor<3xf64>, tensor<3x2xf64>, tensor<3x2xf64>, tensor<3x2x2xf64>
  }
}
