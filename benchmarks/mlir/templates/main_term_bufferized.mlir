// This takes main_term.mlir and bufferizes everything by hand in a hopefully optimal way.
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#view = affine_map<(d0)[s0] -> (d0 + s0)>

//
// Handwritten Main Term Compressed Grad
//
func @bQtimesx(%Ltri_slice: memref<{{tri_size}}xf64, #view>, %xcentered: memref<{{d}}xf64>, %out: memref<{{d}}xf64>) {
  %zero = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cd = arith.constant {{d}} : index
  scf.for %iv = %c0 to %cd step %c1 {
    %Lidx_0 = arith.muli %c2, %cd : index
    %Lidx_1 = arith.subi %Lidx_0, %iv : index
    %Lidx_2 = arith.subi %Lidx_1, %c1 : index
    %Lidx_3 = arith.muli %Lidx_2, %iv : index
    %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

    %iv_plus_1 = arith.addi %iv, %c1 : index
    scf.for %jv = %iv_plus_1 to %cd step %c1 {
      %Lidx_5 = arith.addi %Lidx_4, %jv : index
      %Lidx = arith.subi %Lidx_5, %iv_plus_1 : index
      %0 = memref.load %Ltri_slice[%Lidx] : memref<{{tri_size}}xf64, #view>
      %1 = memref.load %xcentered[%iv] : memref<{{d}}xf64>
      %2 = memref.load %out[%jv] : memref<{{d}}xf64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %3, %2 : f64
      memref.store %4, %out[%jv] : memref<{{d}}xf64>
    }
  }
  return
}

func @bvecmat(%x: memref<{{d}}xf64>, %L: memref<{{tri_size}}xf64, #view>, %out: memref<{{d}}xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cd = arith.constant {{d}} : index
  scf.for %iv = %c0 to %cd step %c1 {
    %Lidx_0 = arith.muli %c2, %cd : index
    %Lidx_1 = arith.subi %Lidx_0, %iv : index
    %Lidx_2 = arith.subi %Lidx_1, %c1 : index
    %Lidx_3 = arith.muli %Lidx_2, %iv : index
    %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

    %iv_plus_1 = arith.addi %iv, %c1 : index
    scf.for %jv = %iv_plus_1 to %cd step %c1 {
      %Lidx_5 = arith.addi %Lidx_4, %jv : index
      %Lidx = arith.subi %Lidx_5, %iv_plus_1 : index
      %0 = memref.load %L[%Lidx] : memref<{{tri_size}}xf64, #view>
      %1 = memref.load %x[%jv] : memref<{{d}}xf64>
      %2 = memref.load %out[%iv] : memref<{{d}}xf64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %3, %2 : f64
      memref.store %4, %out[%iv] : memref<{{d}}xf64>
    }
  }
  return
}

func @bouter_product(%x: memref<{{d}}xf64>, %y: memref<{{d}}xf64, #view>, %out: memref<{{tri_size}}xf64, #view>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cd = arith.constant {{d}} : index
  scf.for %iv = %c0 to %cd step %c1 {
    %Lidx_0 = arith.muli %c2, %cd : index
    %Lidx_1 = arith.subi %Lidx_0, %iv : index
    %Lidx_2 = arith.subi %Lidx_1, %c1 : index
    %Lidx_3 = arith.muli %Lidx_2, %iv : index
    %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

    %iv_plus_1 = arith.addi %iv, %c1 : index
    scf.for %jv = %iv_plus_1 to %cd step %c1 {
      %Lidx_5 = arith.addi %Lidx_4, %jv : index
      %Lidx = arith.subi %Lidx_5, %iv_plus_1 : index
      %0 = memref.load %out[%Lidx] : memref<{{tri_size}}xf64, #view>
      %1 = memref.load %x[%jv] : memref<{{d}}xf64>
      %2 = memref.load %y[%iv] : memref<{{d}}xf64, #view>
      %3 = arith.mulf %1, %2 : f64
      %4 = arith.addf %3, %0 : f64
      memref.store %4, %out[%Lidx] : memref<{{tri_size}}xf64, #view>
    }
  }
  return
}

func private @print_memref_f64(memref<*xf64>) attributes { llvm.emit_c_interface }

func @handwritten_main_term_compressed_buf_grad(
  %alphas: memref<{{k}}xf64>,
  %means: memref<{{k}}x{{d}}xf64>,
  %Qs: memref<{{k}}x{{d}}xf64>,
  %Ls: memref<{{k}}x{{tri_size}}xf64>,
  %x: memref<{{n}}x{{d}}xf64>
) -> (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{tri_size}}xf64>) {
  %zero = arith.constant 0.0 : f64
  %Qdiags = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %sum_qs = memref.alloc() : memref<{{k}}xf64>
  %xcentered = memref.alloc() : memref<{{d}}xf64>
  %Qxcentered = memref.alloc() : memref<{{d}}xf64>
  %main_term = memref.alloc() : memref<{{k}}xf64>

  linalg.generic
    {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
    ins(%Qs : memref<{{k}}x{{d}}xf64>)
    outs(%Qdiags : memref<{{k}}x{{d}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %39 = math.exp %arg7 : f64
    linalg.yield %39 : f64
  }

  linalg.fill(%zero, %sum_qs) : f64, memref<{{k}}xf64>
  linalg.generic
    {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]}
    ins(%Qs : memref<{{k}}x{{d}}xf64>)
    outs(%sum_qs : memref<{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %39 = arith.addf %arg7, %arg8 : f64
    linalg.yield %39 : f64
  }

  %g = arith.constant 1.0 : f64
  %dalphas = memref.alloc() : memref<{{k}}xf64>
  %dmeans = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %dQdiags = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %dsum_qs = memref.alloc() : memref<{{k}}xf64>
  %dxcentered = memref.alloc() : memref<{{d}}xf64>
  %dQxcentered = memref.alloc() : memref<{{d}}xf64>
  %dmain_term = memref.alloc() : memref<{{k}}xf64>
  %dQs = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %dLs = memref.alloc() : memref<{{k}}x{{tri_size}}xf64>
  linalg.fill(%zero, %dalphas) : f64, memref<{{k}}xf64>
  linalg.fill(%zero, %dmeans) : f64, memref<{{k}}x{{d}}xf64>
  linalg.fill(%zero, %dQdiags) : f64, memref<{{k}}x{{d}}xf64>
  linalg.fill(%zero, %dsum_qs) : f64, memref<{{k}}xf64>
  linalg.fill(%zero, %dLs) : f64, memref<{{k}}x{{tri_size}}xf64>
  
  %msqnorm = memref.alloca() : memref<f64>
  %se_space = memref.alloca() : memref<f64>

  %xcentered_cache = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %Qxcentered_cache = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ck = arith.constant {{k}} : index
  %cn = arith.constant {{n}} : index
  %half = arith.constant 0.5 : f64
  scf.for %iv = %c0 to %cn step %c1 {
    scf.for %kv = %c0 to %ck step %c1 {
      %x_slice = memref.subview %x[%iv, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      %means_slice = memref.subview %means[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      linalg.generic
        { indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"] }
        ins(%x_slice, %means_slice : memref<{{d}}xf64, #view>, memref<{{d}}xf64, #view>)
        outs(%xcentered : memref<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.subf %arg0, %arg1 : f64
        linalg.yield %0 : f64
      }
      // cache xcentered
      %xview = memref.subview %xcentered_cache[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      linalg.copy(%xcentered, %xview) : memref<{{d}}xf64>, memref<{{d}}xf64, #view>

      %Qdiags_slice = memref.subview %Qdiags[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      linalg.generic
        { indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"] }
        ins(%Qdiags_slice, %xcentered : memref<{{d}}xf64, #view>, memref<{{d}}xf64>)
        outs(%Qxcentered : memref<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        linalg.yield %0 : f64
      }
      %Ls_slice = memref.subview %Ls[%kv, 0] [1, {{tri_size}}] [1, 1] : memref<{{k}}x{{tri_size}}xf64> to memref<{{tri_size}}xf64, #view>
      call @bQtimesx(%Ls_slice, %xcentered, %Qxcentered) : (memref<{{tri_size}}xf64, #view>, memref<{{d}}xf64>, memref<{{d}}xf64>) -> ()

      // cache Qxcentered
      %qview = memref.subview %Qxcentered_cache[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      linalg.copy(%Qxcentered, %qview) : memref<{{d}}xf64>, memref<{{d}}xf64, #view>

      memref.store %zero, %msqnorm[] : memref<f64>
      linalg.generic
        {
          indexing_maps = [#map2, affine_map<(d0) -> ()>],
          iterator_types = ["reduction"]
        }
        ins(%Qxcentered : memref<{{d}}xf64>)
        outs(%msqnorm : memref<f64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.mulf %arg0, %arg0 : f64
        %1 = arith.addf %0, %arg1 : f64
        linalg.yield %1 : f64
      }
      %msqnorm_val = memref.load %msqnorm[] : memref<f64>

      %hmsqnorm = arith.mulf %half, %msqnorm_val : f64
      %a_k = memref.load %alphas[%kv] : memref<{{k}}xf64>
      %sum_q_k = memref.load %sum_qs[%kv] : memref<{{k}}xf64>
      %mt0 = arith.addf %a_k, %sum_q_k : f64
      %mt1 = arith.subf %mt0, %hmsqnorm : f64
      memref.store %mt1, %main_term[%kv] : memref<{{k}}xf64>
    }

    // grad_logsumexp
    %max_init_val = memref.load %main_term[%c0] : memref<{{k}}xf64>
    memref.store %max_init_val, %msqnorm[] : memref<f64>
    linalg.generic
      { indexing_maps = [#map2, affine_map<(d0) -> ()>], iterator_types = ["reduction"] }
      ins(%main_term : memref<{{k}}xf64>)
      outs(%msqnorm : memref<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %p = arith.cmpf ogt, %arg0, %arg1 : f64
      %0 = select %p, %arg0, %arg1 : f64
      linalg.yield %0 : f64
    }
    %max_val = memref.load %msqnorm[] : memref<f64>

    memref.store %zero, %se_space[] : memref<f64>
    linalg.generic
      { indexing_maps = [#map2, affine_map<(d0) -> ()>], iterator_types = ["reduction"] }
      ins(%main_term : memref<{{k}}xf64>)
      outs(%se_space : memref<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %pre = arith.subf %arg0, %max_val : f64
      %0 = math.exp %pre : f64
      %1 = arith.addf %0, %arg1 : f64
      linalg.yield %1 : f64
    }
    %se_val = memref.load %se_space[] : memref<f64>
    %lse = math.log %se_val : f64
    %logsumexp = arith.addf %lse, %max_val : f64
    linalg.generic
      { indexing_maps = [#map2, #map2], iterator_types = ["parallel"]}
      ins(%main_term : memref<{{k}}xf64>)
      outs(%dmain_term : memref<{{k}}xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.subf %arg0, %logsumexp : f64
      %1 = math.exp %0 : f64
      %2 = arith.mulf %1, %g : f64
      linalg.yield %2 : f64
    }
    // end grad_logsumexp

    linalg.generic
      { indexing_maps = [#map2, #map2], iterator_types = ["parallel"] }
      ins(%dmain_term : memref<{{k}}xf64>)
      outs(%dalphas : memref<{{k}}xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.addf %arg0, %arg1 : f64
      linalg.yield %0 : f64
    }
    linalg.generic
      { indexing_maps = [#map2, #map2], iterator_types = ["parallel"] }
      ins(%dmain_term : memref<{{k}}xf64>)
      outs(%dsum_qs : memref<{{k}}xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.addf %arg0, %arg1 : f64
      linalg.yield %0 : f64
    }

    scf.for %kv = %c0 to %ck step %c1 {
      // Read primal values from cache
      %xcentered_cached = memref.subview %xcentered_cache[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      %Qxcentered_cached = memref.subview %Qxcentered_cache[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>

      %dmsqnorm_0 = memref.load %dmain_term[%kv] : memref<{{k}}xf64>
      %dmsqnorm = arith.negf %dmsqnorm_0 : f64
      linalg.generic
        { indexing_maps = [#map2, #map2], iterator_types = ["parallel"] }
        ins(%Qxcentered_cached : memref<{{d}}xf64, #view>)
        outs(%dQxcentered : memref<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.mulf %arg0, %dmsqnorm : f64
        linalg.yield %0 : f64
      }
      %Qdiags_slice = memref.subview %Qdiags[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      %Ls_slice = memref.subview %Ls[%kv, 0] [1, {{tri_size}}] [1, 1] : memref<{{k}}x{{tri_size}}xf64> to memref<{{tri_size}}xf64, #view>
      linalg.generic
        { doc = "arith.mulf", indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"] }
        ins(%Qdiags_slice, %dQxcentered : memref<{{d}}xf64, #view>, memref<{{d}}xf64>)
        outs(%dxcentered : memref<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        linalg.yield %0 : f64
      }
      call @bvecmat(%dQxcentered, %Ls_slice, %dxcentered) : (memref<{{d}}xf64>, memref<{{tri_size}}xf64, #view>, memref<{{d}}xf64>) -> ()

      %dmeans_slice = memref.subview %dmeans[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      linalg.generic
        { indexing_maps = [#map2, #map2], iterator_types = ["parallel"] }
        ins(%dxcentered : memref<{{d}}xf64>)
        outs(%dmeans_slice : memref<{{d}}xf64, #view>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.subf %arg1, %arg0 : f64
        linalg.yield %0 : f64
      }

      %dQdiags_slice = memref.subview %dQdiags[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      linalg.generic
        { indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"] }
        ins(%dQxcentered, %xcentered_cached : memref<{{d}}xf64>, memref<{{d}}xf64, #view>)
        outs(%dQdiags_slice : memref<{{d}}xf64, #view>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        %1 = arith.addf %0, %arg2 : f64
        linalg.yield %1 : f64
      }

      %dLs_slice = memref.subview %dLs[%kv, 0] [1, {{tri_size}}] [1, 1] : memref<{{k}}x{{tri_size}}xf64> to memref<{{tri_size}}xf64, #view>
      call @bouter_product(%dQxcentered, %xcentered_cached, %dLs_slice) : (memref<{{d}}xf64>, memref<{{d}}xf64, #view>, memref<{{tri_size}}xf64, #view>) -> ()
    }
  }

  linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>, #map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%dsum_qs, %dQdiags, %Qdiags : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>)
    outs(%dQs : memref<{{k}}x{{d}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg1, %arg2 : f64
    %1 = arith.addf %0, %arg0 : f64
    linalg.yield %1 : f64
  }

  return %dalphas, %dmeans, %dQs, %dLs : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{tri_size}}xf64>
}
