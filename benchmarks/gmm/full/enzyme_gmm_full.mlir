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
  func @mQtimesx(%ik: index, %Qdiag: memref<{{k}}x{{d}}xf64>, %ltri: memref<{{k}}x{{d}}x{{d}}xf64>, %x: memref<{{d}}xf64>, %out: memref<{{d}}xf64>) {
    // Elementwise multiplication
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cd = arith.constant {{d}} : index
    scf.for %iv = %c0 to %cd step %c1 {
      %0 = memref.load %Qdiag[%ik, %iv] : memref<{{k}}x{{d}}xf64>
      %1 = memref.load %x[%iv] : memref<{{d}}xf64>
      %2 = arith.mulf %0, %1 : f64
      memref.store %2, %out[%iv] : memref<{{d}}xf64>
    }

    // The triangular matrix-vector multiplication
    scf.for %iv = %c1 to %cd step %c1 {
      scf.for %jv = %c0 to %cd step %c1 {
        %0 = memref.load %ltri[%ik, %iv, %jv] : memref<{{k}}x{{d}}x{{d}}xf64>
        %1 = memref.load %x[%jv] : memref<{{d}}xf64>
        %2 = memref.load %out[%iv] : memref<{{d}}xf64>
        %3 = arith.mulf %0, %1 : f64
        %4 = arith.addf %3, %2 : f64
        memref.store %4, %out[%iv] : memref<{{d}}xf64>
      }
    }
    return
  }

  func @msqnorm(%x : memref<{{d}}xf64>) -> f64 {
    %zero = arith.constant 0.0 : f64
    %out = memref.alloc() : memref<f64>
    memref.store %zero, %out[] : memref<f64>
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%x : memref<{{d}}xf64>) outs(%out : memref<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.mulf %arg0, %arg0 : f64
      %1 = arith.addf %0, %arg1 : f64
      linalg.yield %1 : f64
    }
    %val = memref.load %out[] : memref<f64>
    memref.dealloc %out : memref<f64>
    return %val : f64
  }

  func @msqnormk(%ik: index, %x: memref<{{k}}x{{d}}xf64>) -> f64 {
    %zero = arith.constant 0.0 : f64
    %out = memref.alloc() : memref<f64>
    memref.store %zero, %out[] : memref<f64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cd = arith.constant {{d}} : index
    scf.for %iv = %c0 to %cd step %c1 {
      %0 = memref.load %x[%ik, %iv] : memref<{{k}}x{{d}}xf64>
      %1 = memref.load %out[] : memref<f64>
      %2 = arith.mulf %0, %0 : f64
      %3 = arith.addf %2, %1 : f64
      memref.store %3, %out[] : memref<f64>
    }
    %val = memref.load %out[] : memref<f64>
    memref.dealloc %out : memref<f64>
    return %val : f64
  }

  func @msqnorm_2d(%ik: index, %x: memref<{{k}}x{{d}}x{{d}}xf64>) -> f64 {
    %zero = arith.constant 0.0 : f64
    %out = memref.alloc() : memref<f64>
    memref.store %zero, %out[] : memref<f64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cd = arith.constant {{d}} : index
    scf.for %iv = %c0 to %cd step %c1 {
      scf.for %jv = %c0 to %cd step %c1 {
        %0 = memref.load %x[%ik, %iv, %jv] : memref<{{k}}x{{d}}x{{d}}xf64>
        %1 = memref.load %out[] : memref<f64>
        %2 = arith.mulf %0, %0 : f64
        %3 = arith.addf %1, %2 : f64
        memref.store %3, %out[] : memref<f64>
      }
    }

    %val = memref.load %out[] : memref<f64>
    memref.dealloc %out : memref<f64>
    return %val : f64
  }

  func @mlogsumexp(%x : memref<{{k}}xf64>) -> f64 {
    // find the max
    %max_space = memref.alloc() : memref<f64>
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f64
    %max_init = memref.load %x[%c0] : memref<{{k}}xf64>
    memref.store %max_init, %max_space[] : memref<f64>

    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%x : memref<{{k}}xf64>) outs(%max_space : memref<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %p = arith.cmpf "ogt", %arg0, %arg1 : f64
      %next = select %p, %arg0, %arg1 : f64
      linalg.yield %next : f64
    }

    %max = memref.load %max_space[] : memref<f64>
    memref.store %zero, %max_space[] : memref<f64>
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%x : memref<{{k}}xf64>) outs(%max_space : memref<f64>) {
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

  func @mlog_wishart_prior(%wishart_gamma: f64, %wishart_m: i64, %sum_qs: memref<{{k}}xf64>, %Qdiags: memref<{{k}}x{{d}}xf64>, %Ltri: memref<{{k}}x{{d}}x{{d}}xf64>, %out: memref<f64>) {
    %c0 = arith.constant 0 : index
    %step = arith.constant 1 : index
    %c1 = arith.constant 1 : i64
    %cd = arith.constant {{d}} : i64
    %ck = arith.constant {{k}} : index
    %n_0 = arith.addi %wishart_m, %c1 : i64
    %n = arith.addi %n_0, %cd : i64
    %half = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64

    %val = scf.for %ik = %c0 to %ck step %step iter_args(%out_iter = %zero) -> (f64) {
      %frobenius_0 = call @msqnormk(%ik, %Qdiags) : (index, memref<{{k}}x{{d}}xf64>) -> f64
      %frobenius_1 = call @msqnorm_2d(%ik, %Ltri) : (index, memref<{{k}}x{{d}}x{{d}}xf64>) -> f64
      %frobenius = arith.addf %frobenius_0, %frobenius_1 : f64

      %out_0 = arith.mulf %wishart_gamma, %wishart_gamma : f64
      %out_1 = arith.mulf %out_0, %half : f64
      %out_2 = arith.mulf %out_1, %frobenius : f64
      %out_3 = arith.sitofp %wishart_m : i64 to f64
      %out_4 = memref.load %sum_qs[%ik] : memref<{{k}}xf64>
      %out_5 = arith.mulf %out_3, %out_4 : f64
      %out_6 = arith.subf %out_2, %out_5 : f64
      %out_next = arith.addf %out_iter, %out_6 : f64
      scf.yield %out_next : f64
    }
    memref.store %val, %out[] : memref<f64>
    return
  }

  func private @print_memref_f64(memref<*xf64>) attributes { llvm.emit_c_interface }

  func @emgmm_full(%alphas: memref<{{k}}xf64>, %means: memref<{{k}}x{{d}}xf64>, %Qs: memref<{{k}}x{{d}}xf64>, %Ls: memref<{{k}}x{{d}}x{{d}}xf64>, %x: memref<{{n}}x{{d}}xf64>, %wishart_gamma: f64, %wishart_m: i64) -> f64 {
    %Qdiags = memref.alloc() : memref<{{k}}x{{d}}xf64>
    %sum_qs = memref.alloc() : memref<{{k}}xf64>
    %xcentered = memref.alloc() : memref<{{d}}xf64>
    %Qxcentered = memref.alloc() : memref<{{d}}xf64>
    %main_term = memref.alloc() : memref<{{k}}xf64>

    // This is the preprocess Qs implementation in the original function.
    linalg.generic
      {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%Qs : memref<{{k}}x{{d}}xf64>)
      outs(%Qdiags : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = math.exp %arg7 : f64
      linalg.yield %39 : f64
    }

    %zero = arith.constant 0.0 : f64
    linalg.fill(%zero, %sum_qs) : f64, memref<{{k}}xf64>
    linalg.fill(%zero, %Qxcentered) : f64, memref<{{d}}xf64>
    linalg.generic
      {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]}
      ins(%Qs : memref<{{k}}x{{d}}xf64>)
      outs(%sum_qs : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = arith.addf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }

    %half = arith.constant 0.5 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cn = arith.constant {{n}} : index
    %ck = arith.constant {{k}} : index
    %cd = arith.constant {{d}} : index
    %slse = scf.for %ix = %c0 to %cn step %c1 iter_args(%slse_iv = %zero) -> f64 {
      scf.for %ik = %c0 to %ck step %c1 {
        // %p0 = arith.cmpi "eq", %ix, %c0 : index
        // %p1 = arith.cmpi "eq", %ik, %c1 : index
        // %p = arith.andi %p0, %p1 : i1
        scf.for %iv = %c0 to %cd step %c1 {
          %0 = memref.load %x[%ix, %iv] : memref<{{n}}x{{d}}xf64>
          %1 = memref.load %means[%ik, %iv] : memref<{{k}}x{{d}}xf64>
          %2 = arith.subf %0, %1 : f64
          memref.store %2, %xcentered[%iv] : memref<{{d}}xf64>
        }
        call @mQtimesx(%ik, %Qdiags, %Ls, %xcentered, %Qxcentered) : (index, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>, memref<{{d}}xf64>, memref<{{d}}xf64>) -> ()

        %msqnorm = call @msqnorm(%Qxcentered) : (memref<{{d}}xf64>) -> f64
        %hmsqnorm = arith.mulf %msqnorm, %half : f64
        %a_ik = memref.load %alphas[%ik] : memref<{{k}}xf64>
        %q_ik = memref.load %sum_qs[%ik] : memref<{{k}}xf64>
        %sum_aq = arith.addf %a_ik, %q_ik : f64
        %main_term_ik = arith.subf %sum_aq, %hmsqnorm : f64
        memref.store %main_term_ik, %main_term[%ik] : memref<{{k}}xf64>
      }

      %slse_iter = call @mlogsumexp(%main_term) : (memref<{{k}}xf64>) -> f64
      %slse_next = arith.addf %slse_iv, %slse_iter : f64
      scf.yield %slse_next : f64
    }

    %lse_alphas = call @mlogsumexp(%alphas) : (memref<{{k}}xf64>) -> f64

    %cn_float = arith.constant {{n}}.0 : f64
    %nlse_alphas = arith.mulf %cn_float, %lse_alphas : f64

    %lwishpri_out = memref.alloca() : memref<f64>
    call @mlog_wishart_prior(%wishart_gamma, %wishart_m, %sum_qs, %Qdiags, %Ls, %lwishpri_out) : (f64, i64, memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>, memref<f64>) -> ()
    %lwishpri = memref.load %lwishpri_out[] : memref<f64>

    %final_0 = arith.subf %slse, %nlse_alphas : f64
    %final = arith.addf %final_0, %lwishpri : f64

    memref.dealloc %Qdiags : memref<{{k}}x{{d}}xf64>
    memref.dealloc %sum_qs : memref<{{k}}xf64>
    memref.dealloc %xcentered : memref<{{d}}xf64>
    memref.dealloc %Qxcentered : memref<{{d}}xf64>
    memref.dealloc %main_term : memref<{{k}}xf64>
    return %final : f64
  }

  func @enzyme_mlir_gmm_full(
    %arg0: memref<{{k}}xf64>,
    %arg1: memref<{{k}}x{{d}}xf64>,
    %arg2: memref<{{k}}x{{d}}xf64>,
    %arg3: memref<{{k}}x{{d}}x{{d}}xf64>,
    %arg4: memref<{{n}}x{{d}}xf64>,
    %arg5: f64,
    %arg6: i64
  ) -> (
    memref<{{k}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}x{{d}}xf64>
  ) {
    %zero = arith.constant 0.0 : f64
    %darg0 = memref.alloc() : memref<{{k}}xf64>
    %darg1 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    %darg2 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    %darg3 = memref.alloc() : memref<{{k}}x{{d}}x{{d}}xf64>

    linalg.fill(%zero, %darg0) : f64, memref<{{k}}xf64>
    linalg.fill(%zero, %darg1) : f64, memref<{{k}}x{{d}}xf64>
    linalg.fill(%zero, %darg2) : f64, memref<{{k}}x{{d}}xf64>
    linalg.fill(%zero, %darg3) : f64, memref<{{k}}x{{d}}x{{d}}xf64>

    %f = constant @emgmm_full : (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>, f64, i64) -> f64
    %df = lagrad.diff %f {const = [4]} : (
      memref<{{k}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}x{{d}}xf64>,
      memref<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> f64, (
      memref<{{k}}xf64>,
      memref<{{k}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}x{{d}}xf64>,
      memref<{{k}}x{{d}}x{{d}}xf64>,
      memref<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> f64
    call_indirect %df(%arg0, %darg0, %arg1, %darg1, %arg2, %darg2, %arg3, %darg3, %arg4, %arg5, %arg6) : (
      memref<{{k}}xf64>,
      memref<{{k}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}x{{d}}xf64>,
      memref<{{k}}x{{d}}x{{d}}xf64>,
      memref<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> f64

    return %darg0, %darg1, %darg2, %darg3 : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>
  }
}