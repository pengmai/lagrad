// The purpose of this file is to hand-optimize the memory references used in the
// primal to see if that affects Enzyme's performance.

// The original Enzyme GMM implementation uses 2856x less memory than the naive MLIR
// implementation!

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
  memref.global "private" constant @__constant_xf64_0 : memref<f64> = dense<1.000000e+03>
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1000xf64_0 : memref<1000xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1000xf64 : memref<1000xf64> = dense<-1.000000e+09>
  memref.global "private" constant @__constant_1000x200xf64 : memref<1000x200xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1000x200x128xf64 : memref<1000x200x128xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_200xf64 : memref<200xf64> = dense<0.000000e+00>

  // Elementwise subtraction
  func private @msubtract(%arg0: memref<128xf64>, %arg1: memref<128xf64>, %arg2: memref<128xf64>) {
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<128xf64>, memref<128xf64>) outs(%arg2 : memref<128xf64>) {
    ^bb0(%arg3: f64, %arg4: f64, %arg5: f64):
      %0 = arith.subf %arg3, %arg4 : f64
      linalg.yield %0 : f64
    }
    return
  }

  func private @mQtimesx(%Qdiag: memref<128xf64>, %ltri: memref<128x128xf64>, %x: memref<128xf64>, %out: memref<128xf64>) {
    // Elementwise multiplication
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%Qdiag, %x : memref<128xf64>, memref<128xf64>) outs(%out : memref<128xf64>) {
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      %0 = arith.mulf %arg0, %arg1 : f64
      linalg.yield %0 : f64
    }

    // The triangular matrix-vector multiplication
    linalg.generic {indexing_maps = [#map0, #map10, #map1], iterator_types = ["parallel", "reduction"]} ins(%ltri, %x : memref<128x128xf64>, memref<128xf64>) outs(%out : memref<128xf64>) {
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      %0 = arith.mulf %arg0, %arg1 : f64
      %1 = arith.addf %0, %arg2 : f64
      linalg.yield %1 : f64
    }
    return
  }

  func @msqnorm(%x : memref<?xf64>) -> f64 {
    %out = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%x : memref<?xf64>) outs(%out : memref<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.mulf %arg0, %arg0 : f64
      %1 = arith.addf %0, %arg1 : f64
      linalg.yield %1 : f64
    }
    %val = memref.load %out[] : memref<f64>
    memref.dealloc %out : memref<f64>
    return %val : f64
  }

  func @msqnorm_2d(%x : memref<?x?xf64>) -> f64 {
    %out = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map0, #map14], iterator_types = ["reduction", "reduction"]} ins(%x : memref<?x?xf64>) outs(%out : memref<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.mulf %arg0, %arg0 : f64
      %1 = arith.addf %0, %arg1 : f64
      linalg.yield %1 : f64
    }
    %val = memref.load %out[] : memref<f64>
    memref.dealloc %out : memref<f64>
    return %val : f64
  }

  func @mlogsumexp(%x : memref<200xf64>) -> f64 {
    // find the max
    %max_space = memref.alloc() : memref<f64>
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f64
    %max_init = memref.load %x[%c0] : memref<200xf64>
    memref.store %max_init, %max_space[] : memref<f64>

    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%x : memref<200xf64>) outs(%max_space : memref<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %p = arith.cmpf "ogt", %arg0, %arg1 : f64
      %next = select %p, %arg0, %arg1 : f64
      linalg.yield %next : f64
    }

    %max = memref.load %max_space[] : memref<f64>
    memref.store %zero, %max_space[] : memref<f64>
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%x : memref<200xf64>) outs(%max_space : memref<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.subf %arg0, %max : f64
      %1 = math.exp %0 : f64
      %2 = arith.addf %1, %arg1 : f64
      linalg.yield %2 : f64
    }
    %se_noadd = memref.load %max_space[] : memref<f64>
    memref.dealloc %max_space : memref<f64>
    %lse_noadd = math.log %se_noadd : f64
    %lse = arith.addf %lse_noadd, %max : f64
    return %lse : f64
  }

  func @mlog_wishart_prior(%wishart_gamma: f64, %wishart_m: i64, %sum_qs: memref<200xf64>, %Qdiags: memref<200x128xf64>, %Ltri: memref<200x128x128xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %step = arith.constant 1 : index
    %c1 = arith.constant 1 : i64
    %cd = arith.constant 128 : i64
    %ck = arith.constant 200 : index
    %n_0 = arith.addi %wishart_m, %c1 : i64
    %n = arith.addi %n_0, %cd : i64
    %half = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64

    %out = scf.for %ik = %c0 to %ck step %step iter_args(%out_iter = %zero) -> (f64) {
      %Qdiags_slice = memref.subview %Qdiags[%ik, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64>
      %Qslice_casted = memref.cast %Qdiags_slice : memref<128xf64> to memref<?xf64>
      %frobenius_0 = call @msqnorm(%Qslice_casted) : (memref<?xf64>) -> f64
      %Ltri_slice = memref.subview %Ltri[%ik, 0, 0] [1, 128, 128] [1, 1, 1] : memref<200x128x128xf64> to memref<128x128xf64>
      %Lslice_casted = memref.cast %Ltri_slice : memref<128x128xf64> to memref<?x?xf64>
      %frobenius_1 = call @msqnorm_2d(%Lslice_casted) : (memref<?x?xf64>) -> f64
      %frobenius = arith.addf %frobenius_0, %frobenius_1 : f64

      %out_0 = arith.mulf %wishart_gamma, %wishart_gamma : f64
      %out_1 = arith.mulf %out_0, %half : f64
      %out_2 = arith.mulf %out_1, %frobenius : f64
      %out_3 = arith.sitofp %wishart_m : i64 to f64
      %out_4 = memref.load %sum_qs[%ik] : memref<200xf64>
      %out_5 = arith.mulf %out_3, %out_4 : f64
      %out_6 = arith.subf %out_2, %out_5 : f64
      %out_next = arith.addf %out_iter, %out_6 : f64
      scf.yield %out_next : f64
    }
    return %out : f64
  }

  func @enzyme_gmm_opt_full(%alphas: memref<200xf64>, %means: memref<200x128xf64>, %Qs: memref<200x128xf64>, %Ls: memref<200x128x128xf64>, %x: memref<1000x128xf64>, %wishart_gamma: f64, %wishart_m: i64) -> f64 {
    %Qdiags = memref.alloc() : memref<200x128xf64>
    %sum_qs = memref.alloc() : memref<200xf64>
    %xcentered = memref.alloc() : memref<128xf64>
    %Qxcentered = memref.alloc() : memref<128xf64>
    %main_term = memref.alloc() : memref<200xf64>

    // This is the preprocess Qs implementation in the original function.
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%Qs : memref<200x128xf64>) outs(%Qdiags : memref<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = math.exp %arg7 : f64
      linalg.yield %39 : f64
    }
    %zero = arith.constant 0.0 : f64
    linalg.fill(%zero, %sum_qs) : f64, memref<200xf64>
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%Qs : memref<200x128xf64>) outs(%sum_qs : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = arith.addf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }

    %half = arith.constant 0.5 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant 1000 : index
    %ck = arith.constant 200 : index
    %slse = scf.for %ix = %c0 to %cn step %c1 iter_args(%slse_iv = %zero) -> f64 {
      scf.for %ik = %c0 to %ck step %c1 {
        %x_slice = memref.subview %x[%ix, 0] [1, 128] [1, 1] : memref<1000x128xf64> to memref<128xf64>
        %means_slice = memref.subview %means[%ik, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64>
        call @msubtract(%x_slice, %means_slice, %xcentered) : (memref<128xf64>, memref<128xf64>, memref<128xf64>) -> ()

        %Qdiags_slice = memref.subview %Qdiags[%ik, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64>
        %Ltri_slice = memref.subview %Ls[%ik, 0, 0] [1, 128, 128] [1, 1, 1] : memref<200x128x128xf64> to memref<128x128xf64>
        // call @mQtimesx(%Qdiags_slice, %Ltri_slice, %xcentered, %Qxcentered) : (memref<128xf64>, memref<128x128xf64>, memref<128xf64>, memref<128xf64>) -> ()

        %Qxunknown = memref.cast %Qxcentered : memref<128xf64> to memref<?xf64>
        %msqnorm = call @msqnorm(%Qxunknown) : (memref<?xf64>) -> f64
        %hmsqnorm = arith.mulf %msqnorm, %half : f64
        %a_ik = memref.load %alphas[%ik] : memref<200xf64>
        %q_ik = memref.load %sum_qs[%ik] : memref<200xf64>
        %sum_aq = arith.addf %a_ik, %q_ik : f64
        %main_term_ik = arith.subf %sum_aq, %hmsqnorm : f64
        memref.store %main_term_ik, %main_term[%ik] : memref<200xf64>
      }

      %slse_iter = call @mlogsumexp(%main_term) : (memref<200xf64>) -> f64
      %slse_next = arith.addf %slse_iv, %slse_iter : f64
      scf.yield %slse_next : f64
    }

    %lse_alphas = call @mlogsumexp(%alphas) : (memref<200xf64>) -> f64

    %cn_float = arith.constant 1000.0 : f64
    %nlse_alphas = arith.mulf %cn_float, %lse_alphas : f64

    %lwishpri = call @mlog_wishart_prior(%wishart_gamma, %wishart_m, %sum_qs, %Qdiags, %Ls) : (f64, i64, memref<200xf64>, memref<200x128xf64>, memref<200x128x128xf64>) -> f64

    %final_0 = arith.subf %slse, %nlse_alphas : f64
    %final = arith.addf %final_0, %lwishpri : f64

    memref.dealloc %Qdiags : memref<200x128xf64>
    memref.dealloc %sum_qs : memref<200xf64>
    memref.dealloc %xcentered : memref<128xf64>
    memref.dealloc %Qxcentered : memref<128xf64>
    memref.dealloc %main_term : memref<200xf64>
    return %final : f64
  }

  func @enzyme_gmm_opt_diff_full(
    %arg0: memref<200xf64>,
    %arg1: memref<200x128xf64>,
    %arg2: memref<200x128xf64>,
    %arg3: memref<200x128x128xf64>,
    %arg4: memref<1000x128xf64>,
    %arg5: f64,
    %arg6: i64
  ) -> (
    memref<200xf64>,
    memref<200x128xf64>,
    memref<200x128xf64>,
    memref<200x128x128xf64>
  ) {
    %darg0 = memref.alloc() : memref<200xf64>
    %darg1 = memref.alloc() : memref<200x128xf64>
    %darg2 = memref.alloc() : memref<200x128xf64>
    %darg3 = memref.alloc() : memref<200x128x128xf64>

    %f = constant @enzyme_gmm_opt_full : (memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x128x128xf64>, memref<1000x128xf64>, f64, i64) -> f64
    %df = standalone.diff %f {const = [4]} : (
      memref<200xf64>,
      memref<200x128xf64>,
      memref<200x128xf64>,
      memref<200x128x128xf64>,
      memref<1000x128xf64>,
      f64,
      i64
    ) -> f64, (
      memref<200xf64>,
      memref<200xf64>,
      memref<200x128xf64>,
      memref<200x128xf64>,
      memref<200x128xf64>,
      memref<200x128xf64>,
      memref<200x128x128xf64>,
      memref<200x128x128xf64>,
      memref<1000x128xf64>,
      f64,
      i64
    ) -> f64
    call_indirect %df(%arg0, %darg0, %arg1, %darg1, %arg2, %darg2, %arg3, %darg3, %arg4, %arg5, %arg6) : (
      memref<200xf64>,
      memref<200xf64>,
      memref<200x128xf64>,
      memref<200x128xf64>,
      memref<200x128xf64>,
      memref<200x128xf64>,
      memref<200x128x128xf64>,
      memref<200x128x128xf64>,
      memref<1000x128xf64>,
      f64,
      i64
    ) -> f64
    return %darg0, %darg1, %darg2, %darg3 : memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x128x128xf64>
  }
}
