// Building upon gmm_buf_compressed.mlir, this file attempts to tensorize the
// MLIR translation of the compressed C primal to evaluate its performance.

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
  func @cQtimesx(%Qdiag: tensor<{{d}}xf64>, %ltri: tensor<{{tri_size}}xf64>, %x: tensor<{{d}}xf64>, %out: tensor<{{d}}xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cd = arith.constant {{d}} : index
    // Elementwise multiplication
    %out_0 = linalg.generic
      {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]}
      ins(%Qdiag, %x : tensor<{{d}}xf64>, tensor<{{d}}xf64>)
      outs(%out : tensor<{{d}}xf64>) {
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      %0 = arith.mulf %arg0, %arg1 : f64
      linalg.yield %0 : f64
    } -> tensor<{{d}}xf64>

    %c2 = arith.constant 2 : index
    // The triangular matrix-vector multiplication
    %out_final = scf.for %iv = %c0 to %cd step %c1 iter_args(%out_iter_i = %out_0) -> tensor<{{d}}xf64> {
      %Lidx_0 = arith.muli %c2, %cd : index
      %Lidx_1 = arith.subi %Lidx_0, %iv : index
      %Lidx_2 = arith.subi %Lidx_1, %c1 : index
      %Lidx_3 = arith.muli %Lidx_2, %iv : index
      %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

      %iv_plus_1 = arith.addi %iv, %c1 : index
      %out_iter_i_next:2 = scf.for %jv = %iv_plus_1 to %cd step %c1 iter_args(%Lidx = %Lidx_4, %out_iter = %out_iter_i) -> (index, tensor<{{d}}xf64>) {
        %0 = tensor.extract %ltri[%Lidx] : tensor<{{tri_size}}xf64>
        %1 = tensor.extract %x[%iv] : tensor<{{d}}xf64>
        %2 = tensor.extract %out_iter[%jv] : tensor<{{d}}xf64>
        %3 = arith.mulf %0, %1 : f64
        %4 = arith.addf %3, %2 : f64
        %out_next = tensor.insert %4 into %out_iter[%jv] : tensor<{{d}}xf64>

        %Lidx_next = arith.addi %Lidx, %c1 : index
        scf.yield %Lidx_next, %out_next : index, tensor<{{d}}xf64>
      }
      scf.yield %out_iter_i_next#1 : tensor<{{d}}xf64>
    }

    %outm = memref.buffer_cast %out : memref<{{d}}xf64>
    %out_finalm = memref.buffer_cast %out_final : memref<{{d}}xf64>
    linalg.copy(%out_finalm, %outm) : memref<{{d}}xf64>, memref<{{d}}xf64>
    return
  }

  func @msqnorm(%x : tensor<{{d}}xf64>) -> f64 {
    %out_space = arith.constant dense<0.0> : tensor<f64>
    %out = linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%x : tensor<{{d}}xf64>) outs(%out_space : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.mulf %arg0, %arg0 : f64
      %1 = arith.addf %0, %arg1 : f64
      linalg.yield %1 : f64
    } -> tensor<f64>
    %val = tensor.extract %out[] : tensor<f64>
    return %val : f64
  }

  func @msqnorm_2d(%x: tensor<{{tri_size}}xf64>) -> f64 {
    %out_space = arith.constant dense<0.0> : tensor<f64>
    %out = linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%x : tensor<{{tri_size}}xf64>) outs(%out_space : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.mulf %arg0, %arg0 : f64
      %1 = arith.addf %0, %arg1 : f64
      linalg.yield %1 : f64
    } -> tensor<f64>
    %val = tensor.extract %out[] : tensor<f64>
    return %val : f64
  }

  func @mlogsumexp(%x : tensor<{{k}}xf64>) -> f64 {
    // find the max
    %max_space = linalg.init_tensor [] : tensor<f64>
    %c0 = arith.constant 0 : index
    %max_init_val = tensor.extract %x[%c0] : tensor<{{k}}xf64>
    %max_init = tensor.insert %max_init_val into %max_space[] : tensor<f64>

    %max_t = linalg.generic
      {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
      ins(%x : tensor<{{k}}xf64>)
      outs(%max_init : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %p = arith.cmpf "ogt", %arg0, %arg1 : f64
      %next = select %p, %arg0, %arg1 : f64
      linalg.yield %next : f64
    } -> tensor<f64>

    %max = tensor.extract %max_t[] : tensor<f64>
    %zero = arith.constant dense<0.0> : tensor<f64>
    %se_noadd_t = linalg.generic
      {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
      ins(%x : tensor<{{k}}xf64>)
      outs(%zero : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.subf %arg0, %max : f64
      %1 = math.exp %0 : f64
      %2 = arith.addf %1, %arg1 : f64
      linalg.yield %2 : f64
    } -> tensor<f64>
    %se_noadd = tensor.extract %se_noadd_t[] : tensor<f64>
    %lse_noadd = math.log %se_noadd : f64
    %lse = arith.addf %lse_noadd, %max : f64
    return %lse : f64
  }

  func @mlog_wishart_prior(%wishart_gamma: f64, %wishart_m: i64, %sum_qs: tensor<{{k}}xf64>, %Qdiags: tensor<{{k}}x{{d}}xf64>, %Ltri: tensor<{{k}}x{{tri_size}}xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %step = arith.constant 1 : index
    %c1 = arith.constant 1 : i64
    %cd = arith.constant {{d}} : i64
    %ck = arith.constant {{k}} : index
    %n_0 = arith.addi %wishart_m, %c1 : i64
    %n = arith.addi %n_0, %cd : i64
    %half = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %zerod_tensor = arith.constant dense<0.0> : tensor<f64>

    %val = scf.for %ik = %c0 to %ck step %step iter_args(%out_iter = %zero) -> (f64) {
      %Qdiags_slice = tensor.extract_slice %Qdiags[%ik, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      // Inlined msqnorm
      %fro_0 = linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%Qdiags_slice : tensor<{{d}}xf64>) outs(%zerod_tensor : tensor<f64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.mulf %arg0, %arg0 : f64
        %1 = arith.addf %0, %arg1 : f64
        linalg.yield %1 : f64
      } -> tensor<f64>
      %frobenius_0 = tensor.extract %fro_0[] : tensor<f64>
      // %frobenius_0 = call @msqnorm(%Qdiags_slice) : (tensor<{{d}}xf64>) -> f64

      %Ltri_slice = tensor.extract_slice %Ltri[%ik, 0] [1, {{tri_size}}] [1, 1] : tensor<{{k}}x{{tri_size}}xf64> to tensor<{{tri_size}}xf64>
      // Inlined msqnorm_2d
      %fro_1 = linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%Ltri_slice : tensor<{{tri_size}}xf64>) outs(%zerod_tensor : tensor<f64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.mulf %arg0, %arg0 : f64
        %1 = arith.addf %0, %arg1 : f64
        linalg.yield %1 : f64
      } -> tensor<f64>
      %frobenius_1 = tensor.extract %fro_1[] : tensor<f64>
      // %frobenius_1 = call @msqnorm_2d(%Ltri_slice) : (tensor<{{tri_size}}xf64>) -> f64

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
    return %val : f64
  }

  func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

  func @enzyme_gmm_opt_compressed(%alphas: tensor<{{k}}xf64>, %means: tensor<{{k}}x{{d}}xf64>, %Qs: tensor<{{k}}x{{d}}xf64>, %Ls: tensor<{{k}}x{{tri_size}}xf64>, %x: tensor<{{n}}x{{d}}xf64>, %wishart_gamma: f64, %wishart_m: i64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %Qdiags_space = linalg.init_tensor [{{k}}, {{d}}] : tensor<{{k}}x{{d}}xf64>
    %sum_qs_space = arith.constant dense<0.0> : tensor<{{k}}xf64>
    %xcentered_space = linalg.init_tensor [{{d}}] : tensor<{{d}}xf64>
    // %Qxcentered = linalg.init_tensor [{{d}}] : tensor<{{d}}xf64>
    %main_term_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>

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
          outs(%xcentered_space : tensor<{{d}}xf64>) {
        ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
          %0 = arith.subf %arg0, %arg1 : f64
          linalg.yield %0 : f64
        } -> tensor<{{d}}xf64>

        %Qdiags_slice = tensor.extract_slice %Qdiags[%ik, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
        %Ltri_slice = tensor.extract_slice %Ls[%ik, 0] [1, {{tri_size}}] [1, 1] : tensor<{{k}}x{{tri_size}}xf64> to tensor<{{tri_size}}xf64>

        // inlined Qtimesx
        // Elementwise multiplication
        %out_0 = arith.mulf %Qdiags_slice, %xcentered : tensor<{{d}}xf64>

        // The triangular matrix-vector multiplication
        %Qxcentered = scf.for %iv = %c0 to %cd step %c1 iter_args(%out_iter_i = %out_0) -> tensor<{{d}}xf64> {
          %Lidx_0 = arith.muli %c2, %cd : index
          %Lidx_1 = arith.subi %Lidx_0, %iv : index
          %Lidx_2 = arith.subi %Lidx_1, %c1 : index
          %Lidx_3 = arith.muli %Lidx_2, %iv : index
          %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

          %iv_plus_1 = arith.addi %iv, %c1 : index
          %out_iter_i_next:2 = scf.for %jv = %iv_plus_1 to %cd step %c1 iter_args(%Lidx = %Lidx_4, %out_iter = %out_iter_i) -> (index, tensor<{{d}}xf64>) {
            %0 = tensor.extract %Ltri_slice[%Lidx] : tensor<{{tri_size}}xf64>
            %1 = tensor.extract %xcentered[%iv] : tensor<{{d}}xf64>
            %2 = tensor.extract %out_iter[%jv] : tensor<{{d}}xf64>
            %3 = arith.mulf %0, %1 : f64
            %4 = arith.addf %3, %2 : f64
            %out_next = tensor.insert %4 into %out_iter[%jv] : tensor<{{d}}xf64>

            %Lidx_next = arith.addi %Lidx, %c1 : index
            scf.yield %Lidx_next, %out_next : index, tensor<{{d}}xf64>
          }
          scf.yield %out_iter_i_next#1 : tensor<{{d}}xf64>
        }
        // call @cQtimesx(%Qdiags_slice, %Ltri_slice, %xcentered, %Qxcentered) : (tensor<{{d}}xf64>, tensor<{{tri_size}}xf64>, tensor<{{d}}xf64>, tensor<{{d}}xf64>) -> ()

        %msqnorm = call @msqnorm(%Qxcentered) : (tensor<{{d}}xf64>) -> f64
        %hmsqnorm = arith.mulf %msqnorm, %half : f64
        %a_ik = tensor.extract %alphas[%ik] : tensor<{{k}}xf64>
        %q_ik = tensor.extract %sum_qs[%ik] : tensor<{{k}}xf64>
        %sum_aq = arith.addf %a_ik, %q_ik : f64
        %main_term_ik = arith.subf %sum_aq, %hmsqnorm : f64
        %main_term_next = tensor.insert %main_term_ik into %mt_iter[%ik] : tensor<{{k}}xf64>
        scf.yield %main_term_next : tensor<{{k}}xf64>
      }

      %slse_iter = call @mlogsumexp(%main_term) : (tensor<{{k}}xf64>) -> f64
      %slse_next = arith.addf %slse_iv, %slse_iter : f64
      scf.yield %slse_next : f64
    }

    %lse_alphas = call @mlogsumexp(%alphas) : (tensor<{{k}}xf64>) -> f64

    %cn_float = arith.constant {{n}}.0 : f64
    %nlse_alphas = arith.mulf %cn_float, %lse_alphas : f64

    %lwishpri = call @mlog_wishart_prior(%wishart_gamma, %wishart_m, %sum_qs, %Qdiags, %Ls) : (f64, i64, tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{tri_size}}xf64>) -> f64

    %final_0 = arith.subf %slse, %nlse_alphas : f64
    %final = arith.addf %final_0, %lwishpri : f64
    return %final : f64
  }

  // {% if method == 'enzyme_mlir_compressed' %}
  // func @enzyme_gmm_opt_diff_compressed(
  //   %arg0: memref<{{k}}xf64>,
  //   %arg1: memref<{{k}}x{{d}}xf64>,
  //   %arg2: memref<{{k}}x{{d}}xf64>,
  //   %arg3: memref<{{k}}x{{tri_size}}xf64>,
  //   %arg4: memref<{{n}}x{{d}}xf64>,
  //   %arg5: f64,
  //   %arg6: i64
  // ) -> (
  //   memref<{{k}}xf64>,
  //   memref<{{k}}x{{d}}xf64>,
  //   memref<{{k}}x{{d}}xf64>,
  //   memref<{{k}}x{{tri_size}}xf64>
  // ) {
  //   %zero = arith.constant 0.0 : f64
  //   %darg0 = memref.alloc() : memref<{{k}}xf64>
  //   %darg1 = memref.alloc() : memref<{{k}}x{{d}}xf64>
  //   %darg2 = memref.alloc() : memref<{{k}}x{{d}}xf64>
  //   %darg3 = memref.alloc() : memref<{{k}}x{{tri_size}}xf64>

  //   linalg.fill(%zero, %darg0) : f64, memref<{{k}}xf64>
  //   linalg.fill(%zero, %darg1) : f64, memref<{{k}}x{{d}}xf64>
  //   linalg.fill(%zero, %darg2) : f64, memref<{{k}}x{{d}}xf64>
  //   linalg.fill(%zero, %darg3) : f64, memref<{{k}}x{{tri_size}}xf64>

  //   %f = constant @enzyme_gmm_opt_compressed : (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{tri_size}}xf64>, memref<{{n}}x{{d}}xf64>, f64, i64) -> f64
  //   %df = standalone.diff %f {const = [4]} : (
  //     memref<{{k}}xf64>,
  //     memref<{{k}}x{{d}}xf64>,
  //     memref<{{k}}x{{d}}xf64>,
  //     memref<{{k}}x{{tri_size}}xf64>,
  //     memref<{{n}}x{{d}}xf64>,
  //     f64,
  //     i64
  //   ) -> f64, (
  //     memref<{{k}}xf64>,
  //     memref<{{k}}xf64>,
  //     memref<{{k}}x{{d}}xf64>,
  //     memref<{{k}}x{{d}}xf64>,
  //     memref<{{k}}x{{d}}xf64>,
  //     memref<{{k}}x{{d}}xf64>,
  //     memref<{{k}}x{{tri_size}}xf64>,
  //     memref<{{k}}x{{tri_size}}xf64>,
  //     memref<{{n}}x{{d}}xf64>,
  //     f64,
  //     i64
  //   ) -> f64
  //   call_indirect %df(%arg0, %darg0, %arg1, %darg1, %arg2, %darg2, %arg3, %darg3, %arg4, %arg5, %arg6) : (
  //     memref<{{k}}xf64>,
  //     memref<{{k}}xf64>,
  //     memref<{{k}}x{{d}}xf64>,
  //     memref<{{k}}x{{d}}xf64>,
  //     memref<{{k}}x{{d}}xf64>,
  //     memref<{{k}}x{{d}}xf64>,
  //     memref<{{k}}x{{tri_size}}xf64>,
  //     memref<{{k}}x{{tri_size}}xf64>,
  //     memref<{{n}}x{{d}}xf64>,
  //     f64,
  //     i64
  //   ) -> f64
  //   return %darg0, %darg1, %darg2, %darg3 : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{tri_size}}xf64>
  // }
  // {% endif %}
}