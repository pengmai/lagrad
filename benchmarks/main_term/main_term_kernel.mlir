#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#view = affine_map<(d0)[s0] -> (d0 + s0)>


func @grad_logsumexp(%x: tensor<{{d}}xf64>, %g: f64) -> tensor<{{d}}xf64> {
  %zero = arith.constant 0.0 : f64
  %se_space = linalg.init_tensor [] : tensor<f64>

  %c0 = arith.constant 0 : index
  %max_init_val = tensor.extract %x[%c0] : tensor<{{d}}xf64>
  %max_init = tensor.insert %max_init_val into %se_space[] : tensor<f64>
  %max = linalg.generic
    { indexing_maps = [#map2, affine_map<(d0) -> ()>], iterator_types = ["reduction"] }
    ins(%x : tensor<{{d}}xf64>)
    outs(%max_init : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %p = arith.cmpf ogt, %arg0, %arg1 : f64
    %0 = select %p, %arg0, %arg1 : f64
    linalg.yield %0 : f64
  } -> tensor<f64>
  %max_val = tensor.extract %max[] : tensor<f64>

  %se_init = linalg.fill(%zero, %se_space) : f64, tensor<f64> -> tensor<f64>
  %sumexp = linalg.generic
    { indexing_maps = [#map2, affine_map<(d0) -> ()>], iterator_types = ["reduction"] }
    ins(%x : tensor<{{d}}xf64>)
    outs(%se_init : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %pre = arith.subf %arg0, %max_val : f64
    %0 = math.exp %pre : f64
    %1 = arith.addf %0, %arg1 : f64
    linalg.yield %1 : f64
  } -> tensor<f64>
  %se_val = tensor.extract %sumexp[] : tensor<f64>
  %lse = math.log %se_val : f64
  %logsumexp = arith.addf %lse, %max_val : f64
  %dx_space = linalg.init_tensor [{{d}}] : tensor<{{d}}xf64>
  %dx = linalg.generic
    { indexing_maps = [#map2, #map2], iterator_types = ["parallel"]}
    ins(%x : tensor<{{d}}xf64>)
    outs(%dx_space : tensor<{{d}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = arith.subf %arg0, %logsumexp : f64
    %1 = math.exp %0 : f64
    %2 = arith.mulf %1, %g : f64
    linalg.yield %2 : f64
  } -> tensor<{{d}}xf64>
  return %dx : tensor<{{d}}xf64>
}

func @handwritten_main_term_grad(
  %alphas: tensor<{{k}}xf64>,
  %means: tensor<{{k}}x{{d}}xf64>,
  %Qs: tensor<{{k}}x{{d}}xf64>,
  %Ls: tensor<{{k}}x{{d}}x{{d}}xf64>,
  %x: tensor<{{n}}x{{d}}xf64>
) -> (tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>) {
  %zero = arith.constant 0.0 : f64
  %zerot = arith.constant dense<0.0> : tensor<{{d}}xf64>
  %Qdiags_space = linalg.init_tensor [{{k}}, {{d}}] : tensor<{{k}}x{{d}}xf64>
  %Qdiags = linalg.generic
    {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
    ins(%Qs : tensor<{{k}}x{{d}}xf64>)
    outs(%Qdiags_space : tensor<{{k}}x{{d}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %39 = math.exp %arg7 : f64
    linalg.yield %39 : f64
  } -> tensor<{{k}}x{{d}}xf64>

  %sum_qs_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>
  %sum_qs_init = linalg.fill(%zero, %sum_qs_space) : f64, tensor<{{k}}xf64> -> tensor<{{k}}xf64>
  %sum_qs = linalg.generic
    {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]}
    ins(%Qs : tensor<{{k}}x{{d}}xf64>)
    outs(%sum_qs_init : tensor<{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %39 = arith.addf %arg7, %arg8 : f64
    linalg.yield %39 : f64
  } -> tensor<{{k}}xf64>

  %main_term_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>
  %g = arith.constant 1.0 : f64
  %dalphas_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>
  %dalphas = linalg.fill(%zero, %dalphas_space) : f64, tensor<{{k}}xf64> -> tensor<{{k}}xf64>
  %dmeans_space = linalg.init_tensor [{{k}}, {{d}}] : tensor<{{k}}x{{d}}xf64>
  %dmeans = linalg.fill(%zero, %dmeans_space) : f64, tensor<{{k}}x{{d}}xf64> -> tensor<{{k}}x{{d}}xf64>
  %dQdiags_space = linalg.init_tensor [{{k}}, {{d}}] : tensor<{{k}}x{{d}}xf64>
  %dQdiags = linalg.fill(%zero, %dQdiags_space) : f64, tensor<{{k}}x{{d}}xf64> -> tensor<{{k}}x{{d}}xf64>
  %dsum_qs_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>
  %dsum_qs = linalg.fill(%zero, %dsum_qs_space) : f64, tensor<{{k}}xf64> -> tensor<{{k}}xf64>
  %dLs_space = linalg.init_tensor [{{k}}, {{d}}, {{d}}] : tensor<{{k}}x{{d}}x{{d}}xf64>
  %dLs = linalg.fill(%zero, %dLs_space) : f64, tensor<{{k}}x{{d}}x{{d}}xf64> -> tensor<{{k}}x{{d}}x{{d}}xf64>
  %xcentered_space = linalg.init_tensor [{{d}}] : tensor<{{d}}xf64>
  // The computation of Qxcentered will probably result in an additional allocation
  %Qxcentered_space = linalg.init_tensor [{{d}}] : tensor<{{d}}xf64>
  %msqnorm_space = linalg.init_tensor [] : tensor<f64>
  %msqnorm_init = linalg.fill(%zero, %msqnorm_space) : f64, tensor<f64> -> tensor<f64>
  %se_space = linalg.init_tensor [] : tensor<f64>
  %dmain_term_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>

  %xcentered_cache = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %Qxcentered_cache = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ck = arith.constant {{k}} : index
  %cn = arith.constant {{n}} : index
  %half = arith.constant 0.5 : f64
  %res:5 = scf.for %iv = %c0 to %cn step %c1 iter_args(%dalphas_iter = %dalphas, %dsum_qs_iter = %dsum_qs, %dmeans_outer = %dmeans, %dQdiags_outer = %dQdiags, %dLs_outer = %dLs) -> (tensor<{{k}}xf64>, tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>) {
    %main_term = scf.for %kv = %c0 to %ck step %c1 iter_args(%mt_iter = %main_term_space) -> tensor<{{k}}xf64> {
      %x_slice = tensor.extract_slice %x[%iv, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
      %means_slice = tensor.extract_slice %means[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %xcentered = linalg.generic
        { indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"] }
        ins(%x_slice, %means_slice : tensor<{{d}}xf64>, tensor<{{d}}xf64>)
        outs(%xcentered_space : tensor<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.subf %arg0, %arg1 : f64
        linalg.yield %0 : f64
      } -> tensor<{{d}}xf64>
      // cache xcentered
      %xcentered_m = memref.buffer_cast %xcentered : memref<{{d}}xf64>
      %xview = memref.subview %xcentered_cache[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      linalg.copy(%xcentered_m, %xview) : memref<{{d}}xf64>, memref<{{d}}xf64, #view>

      %Qdiags_slice = tensor.extract_slice %Qdiags[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %out0 = linalg.generic
        { indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"] }
        ins(%Qdiags_slice, %xcentered : tensor<{{d}}xf64>, tensor<{{d}}xf64>)
        outs(%Qxcentered_space : tensor<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        linalg.yield %0 : f64
      } -> tensor<{{d}}xf64>
      %Ls_slice = tensor.extract_slice %Ls[%kv, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : tensor<{{k}}x{{d}}x{{d}}xf64> to tensor<{{d}}x{{d}}xf64>
      // %out1 = linalg.generic
      //   {
      //     indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>],
      //     iterator_types = ["parallel", "reduction"]
      //   }
      //   ins(%Ls_slice, %xcentered : tensor<{{d}}x{{d}}xf64>, tensor<{{d}}xf64>) outs(%zerot : tensor<{{d}}xf64>) {
      // ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      //   %0 = arith.mulf %arg0, %arg1 : f64
      //   %1 = arith.addf %0, %arg2 : f64
      //   linalg.yield %1 : f64
      // } -> tensor<{{d}}xf64>
      %Qxcentered = linalg.matvec ins(%Ls_slice, %xcentered : tensor<{{d}}x{{d}}xf64>, tensor<{{d}}xf64>) outs(%out0 : tensor<{{d}}xf64>) -> tensor<{{d}}xf64>
      // %Qxcentered = arith.addf %out1, %out1 : tensor<{{d}}xf64>

      // cache Qxcentered
      %Qxcentered_m = memref.buffer_cast %Qxcentered : memref<{{d}}xf64>
      %qview = memref.subview %Qxcentered_cache[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      linalg.copy(%Qxcentered_m, %qview) : memref<{{d}}xf64>, memref<{{d}}xf64, #view>

      %msqnorm = linalg.generic
        {
          indexing_maps = [#map2, affine_map<(d0) -> ()>],
          iterator_types = ["reduction"]
        }
        ins(%Qxcentered : tensor<{{d}}xf64>)
        outs(%msqnorm_init : tensor<f64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.mulf %arg0, %arg0 : f64
        %1 = arith.addf %0, %arg1 : f64
        linalg.yield %1 : f64
      } -> tensor<f64>
      %msqnorm_val = tensor.extract %msqnorm[] : tensor<f64>
      %hmsqnorm = arith.mulf %half, %msqnorm_val : f64
      %a_k = tensor.extract %alphas[%kv] : tensor<{{k}}xf64>
      %sum_q_k = tensor.extract %sum_qs[%kv] : tensor<{{k}}xf64>
      %mt0 = arith.addf %a_k, %sum_q_k : f64
      %mt1 = arith.subf %mt0, %hmsqnorm : f64
      %mt_next = tensor.insert %mt1 into %mt_iter[%kv] : tensor<{{k}}xf64>
      scf.yield %mt_next : tensor<{{k}}xf64>
    }

    // grad_logsumexp
    %max_init_val = tensor.extract %main_term[%c0] : tensor<{{k}}xf64>
    %max_init = tensor.insert %max_init_val into %se_space[] : tensor<f64>
    %max = linalg.generic
      { indexing_maps = [#map2, affine_map<(d0) -> ()>], iterator_types = ["reduction"] }
      ins(%main_term : tensor<{{k}}xf64>)
      outs(%max_init : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %p = arith.cmpf ogt, %arg0, %arg1 : f64
      %0 = select %p, %arg0, %arg1 : f64
      linalg.yield %0 : f64
    } -> tensor<f64>
    %max_val = tensor.extract %max[] : tensor<f64>

    %se_init = linalg.fill(%zero, %se_space) : f64, tensor<f64> -> tensor<f64>
    %sumexp = linalg.generic
      { indexing_maps = [#map2, affine_map<(d0) -> ()>], iterator_types = ["reduction"] }
      ins(%main_term : tensor<{{k}}xf64>)
      outs(%se_init : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %pre = arith.subf %arg0, %max_val : f64
      %0 = math.exp %pre : f64
      %1 = arith.addf %0, %arg1 : f64
      linalg.yield %1 : f64
    } -> tensor<f64>
    %se_val = tensor.extract %sumexp[] : tensor<f64>
    %lse = math.log %se_val : f64
    %logsumexp = arith.addf %lse, %max_val : f64
    %dmain_term = linalg.generic
      { indexing_maps = [#map2, #map2], iterator_types = ["parallel"]}
      ins(%main_term : tensor<{{k}}xf64>)
      outs(%dmain_term_space : tensor<{{k}}xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.subf %arg0, %logsumexp : f64
      %1 = math.exp %0 : f64
      %2 = arith.mulf %1, %g : f64
      linalg.yield %2 : f64
    } -> tensor<{{k}}xf64>
    // end grad_logsumexp

    %dalphas_next = linalg.generic
      { indexing_maps = [#map2, #map2], iterator_types = ["parallel"] }
      ins(%dmain_term : tensor<{{k}}xf64>)
      outs(%dalphas_iter : tensor<{{k}}xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.addf %arg0, %arg1 : f64
      linalg.yield %0 : f64
    } -> tensor<{{k}}xf64>
    %dsum_qs_next = linalg.generic
      { indexing_maps = [#map2, #map2], iterator_types = ["parallel"] }
      ins(%dmain_term : tensor<{{k}}xf64>)
      outs(%dsum_qs_iter : tensor<{{k}}xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.addf %arg0, %arg1 : f64
      linalg.yield %0 : f64
    } -> tensor<{{k}}xf64>

    %inner_next:3 = scf.for %kv = %c0 to %ck step %c1 iter_args(%dmeans_iter = %dmeans_outer, %dQdiags_iter = %dQdiags_outer, %dLs_iter = %dLs_outer) -> (tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>) {
      // Read primal values from cache
      %xcentered_view = memref.subview %xcentered_cache[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      %xcentered_casted = memref.cast %xcentered_view : memref<{{d}}xf64, #view> to memref<{{d}}xf64>
      %xcentered = memref.tensor_load %xcentered_casted : memref<{{d}}xf64>
      %Qxcentered_view = memref.subview %Qxcentered_cache[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      %Qxcentered_casted = memref.cast %Qxcentered_view : memref<{{d}}xf64, #view> to memref<{{d}}xf64>
      %Qxcentered = memref.tensor_load %Qxcentered_casted : memref<{{d}}xf64>

      %dmsqnorm_0 = tensor.extract %dmain_term[%kv] : tensor<{{k}}xf64>
      %dmsqnorm = arith.negf %dmsqnorm_0 : f64
      %dQxcentered_space = linalg.init_tensor [{{d}}] : tensor<{{d}}xf64>
      %dQxcentered = linalg.generic
        { indexing_maps = [#map2, #map2], iterator_types = ["parallel"] }
        ins(%Qxcentered : tensor<{{d}}xf64>)
        outs(%dQxcentered_space : tensor<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.mulf %arg0, %dmsqnorm : f64
        linalg.yield %0 : f64
      } -> tensor<{{d}}xf64>
      %Qdiags_slice = tensor.extract_slice %Qdiags[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %Ls_slice = tensor.extract_slice %Ls[%kv, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : tensor<{{k}}x{{d}}x{{d}}xf64> to tensor<{{d}}x{{d}}xf64>
      %dxcentered_0 = arith.mulf %Qdiags_slice, %dQxcentered : tensor<{{d}}xf64>
      %dxcentered = linalg.vecmat ins(%dQxcentered, %Ls_slice : tensor<{{d}}xf64>, tensor<{{d}}x{{d}}xf64>) outs(%dxcentered_0 : tensor<{{d}}xf64>) -> tensor<{{d}}xf64>

      %dmeans_slice = tensor.extract_slice %dmeans_iter[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %dmeans_slice_next = linalg.generic
        { indexing_maps = [#map2, #map2], iterator_types = ["parallel"] }
        ins(%dxcentered : tensor<{{d}}xf64>)
        outs(%dmeans_slice : tensor<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.subf %arg1, %arg0 : f64
        linalg.yield %0 : f64
      } -> tensor<{{d}}xf64>
      %dmeans_next = tensor.insert_slice %dmeans_slice_next into %dmeans_iter[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{d}}xf64> into tensor<{{k}}x{{d}}xf64>

      %dQdiags_slice = tensor.extract_slice %dQdiags_iter[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %dQdiags_slice_next = linalg.generic
        { indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"] }
        ins(%dQxcentered, %xcentered : tensor<{{d}}xf64>, tensor<{{d}}xf64>)
        outs(%dQdiags_slice : tensor<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        %1 = arith.addf %0, %arg2 : f64
        linalg.yield %1 : f64
      } -> tensor<{{d}}xf64>
      %dQdiags_next = tensor.insert_slice %dQdiags_slice_next into %dQdiags_iter[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{d}}xf64> into tensor<{{k}}x{{d}}xf64>

      %dLs_slice = tensor.extract_slice %dLs_iter[%kv, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : tensor<{{k}}x{{d}}x{{d}}xf64> to tensor<{{d}}x{{d}}xf64>
      %dLs_slice_next = linalg.generic
        {
          doc = "Vector-vector outer product",
          indexing_maps = [
            affine_map<(d0, d1) -> (d0)>,
            affine_map<(d0, d1) -> (d1)>,
            #map0
          ],
          iterator_types = ["parallel", "parallel"]
        }
        ins(%dQxcentered, %xcentered : tensor<{{d}}xf64>, tensor<{{d}}xf64>)
        outs(%dLs_slice : tensor<{{d}}x{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        %1 = arith.addf %0, %arg2 : f64
        linalg.yield %1 : f64
      } -> tensor<{{d}}x{{d}}xf64>
      %dLs_next = tensor.insert_slice %dLs_slice_next into %dLs_iter[%kv, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : tensor<{{d}}x{{d}}xf64> into tensor<{{k}}x{{d}}x{{d}}xf64>
      scf.yield %dmeans_next, %dQdiags_next, %dLs_next : tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>
    }
    scf.yield %dalphas_next, %dsum_qs_next, %inner_next#0, %inner_next#1, %inner_next#2 : tensor<{{k}}xf64>, tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>
  }

  %dQs_space = linalg.init_tensor [{{k}}, {{d}}] : tensor<{{k}}x{{d}}xf64>
  %dQs = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>, #map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%res#1, %res#3, %Qdiags : tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>)
    outs(%dQs_space : tensor<{{k}}x{{d}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg1, %arg2 : f64
    %1 = arith.addf %0, %arg0 : f64
    linalg.yield %1 : f64
  } -> tensor<{{k}}x{{d}}xf64>

  return %res#0, %res#2, %dQs, %res#4 : tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>
}

//
// Handwritten Main Term Compressed Grad
//
func @Qtimesx(%Ltri_slice: tensor<{{tri_size}}xf64>, %xcentered: tensor<{{d}}xf64>) -> tensor<{{d}}xf64> {
  %zero = arith.constant 0.0 : f64
  %trmv_space = linalg.init_tensor [{{d}}] : tensor<{{d}}xf64>
  %trmv_init = linalg.fill(%zero, %trmv_space) : f64, tensor<{{d}}xf64> -> tensor<{{d}}xf64>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cd = arith.constant {{d}} : index
  %out_1 = scf.for %iv = %c0 to %cd step %c1 iter_args(%out_iter_i = %trmv_init) -> tensor<{{d}}xf64> {
    %Lidx_0 = arith.muli %c2, %cd : index
    %Lidx_1 = arith.subi %Lidx_0, %iv : index
    %Lidx_2 = arith.subi %Lidx_1, %c1 : index
    %Lidx_3 = arith.muli %Lidx_2, %iv : index
    %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

    %iv_plus_1 = arith.addi %iv, %c1 : index
    %out_iter_i_next = scf.for %jv = %iv_plus_1 to %cd step %c1 iter_args(%out_iter = %out_iter_i) -> (tensor<{{d}}xf64>) {
      %Lidx_5 = arith.addi %Lidx_4, %jv : index
      %Lidx = arith.subi %Lidx_5, %iv_plus_1 : index
      %0 = tensor.extract %Ltri_slice[%Lidx] : tensor<{{tri_size}}xf64>
      %1 = tensor.extract %xcentered[%iv] : tensor<{{d}}xf64>
      %2 = tensor.extract %out_iter[%jv] : tensor<{{d}}xf64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %3, %2 : f64
      %out_next = tensor.insert %4 into %out_iter[%jv] : tensor<{{d}}xf64>

      scf.yield %out_next : tensor<{{d}}xf64>
    }
    scf.yield %out_iter_i_next : tensor<{{d}}xf64>
  }
  return %out_1 : tensor<{{d}}xf64>
}

func @vecmat(%x: tensor<{{d}}xf64>, %L: tensor<{{tri_size}}xf64>, %vm_init: tensor<{{d}}xf64>) -> tensor<{{d}}xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cd = arith.constant {{d}} : index
  %out = scf.for %iv = %c0 to %cd step %c1 iter_args(%outer = %vm_init) -> tensor<{{d}}xf64> {
    %Lidx_0 = arith.muli %c2, %cd : index
    %Lidx_1 = arith.subi %Lidx_0, %iv : index
    %Lidx_2 = arith.subi %Lidx_1, %c1 : index
    %Lidx_3 = arith.muli %Lidx_2, %iv : index
    %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

    %iv_plus_1 = arith.addi %iv, %c1 : index
    %outer_next = scf.for %jv = %iv_plus_1 to %cd step %c1 iter_args(%out_iter = %outer) -> (tensor<{{d}}xf64>) {
      %Lidx_5 = arith.addi %Lidx_4, %jv : index
      %Lidx = arith.subi %Lidx_5, %iv_plus_1 : index
      %0 = tensor.extract %L[%Lidx] : tensor<{{tri_size}}xf64>
      %1 = tensor.extract %x[%jv] : tensor<{{d}}xf64>
      %2 = tensor.extract %out_iter[%iv] : tensor<{{d}}xf64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %3, %2 : f64
      %out_next = tensor.insert %4 into %out_iter[%iv] : tensor<{{d}}xf64>
      scf.yield %out_next : tensor<{{d}}xf64>
    }
    scf.yield %outer_next : tensor<{{d}}xf64>
  }
  return %out : tensor<{{d}}xf64>
}

func @outer_product(%x: tensor<{{d}}xf64>, %y: tensor<{{d}}xf64>, %out_init: tensor<{{tri_size}}xf64>) -> tensor<{{tri_size}}xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cd = arith.constant {{d}} : index
  %out = scf.for %iv = %c0 to %cd step %c1 iter_args(%outer = %out_init) -> tensor<{{tri_size}}xf64> {
    %Lidx_0 = arith.muli %c2, %cd : index
    %Lidx_1 = arith.subi %Lidx_0, %iv : index
    %Lidx_2 = arith.subi %Lidx_1, %c1 : index
    %Lidx_3 = arith.muli %Lidx_2, %iv : index
    %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

    %iv_plus_1 = arith.addi %iv, %c1 : index
    %outer_next = scf.for %jv = %iv_plus_1 to %cd step %c1 iter_args(%out_iter = %outer) -> (tensor<{{tri_size}}xf64>) {
      %Lidx_5 = arith.addi %Lidx_4, %jv : index
      %Lidx = arith.subi %Lidx_5, %iv_plus_1 : index
      %0 = tensor.extract %out_iter[%Lidx] : tensor<{{tri_size}}xf64>
      %1 = tensor.extract %x[%jv] : tensor<{{d}}xf64>
      %2 = tensor.extract %y[%iv] : tensor<{{d}}xf64>
      %3 = arith.mulf %1, %2 : f64
      %4 = arith.addf %3, %0 : f64
      %out_next = tensor.insert %4 into %out_iter[%Lidx] : tensor<{{tri_size}}xf64>
      scf.yield %out_next : tensor<{{tri_size}}xf64>
    }
    scf.yield %outer_next : tensor<{{tri_size}}xf64>
  }
  return %out : tensor<{{tri_size}}xf64>
}

// func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @handwritten_main_term_compressed_grad(
  %alphas: tensor<{{k}}xf64>,
  %means: tensor<{{k}}x{{d}}xf64>,
  %Qs: tensor<{{k}}x{{d}}xf64>,
  %Ls: tensor<{{k}}x{{tri_size}}xf64>,
  %x: tensor<{{n}}x{{d}}xf64>
) -> (tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{tri_size}}xf64>) {
  %zero = arith.constant 0.0 : f64
  %Qdiags_space = linalg.init_tensor [{{k}}, {{d}}] : tensor<{{k}}x{{d}}xf64>
  %sum_qs_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>
  %xcentered_space = linalg.init_tensor [{{d}}] : tensor<{{d}}xf64>
  %Qxcentered_space = linalg.init_tensor [{{d}}] : tensor<{{d}}xf64>
  %main_term_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>

  %Qdiags = linalg.generic
    {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
    ins(%Qs : tensor<{{k}}x{{d}}xf64>)
    outs(%Qdiags_space : tensor<{{k}}x{{d}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %39 = math.exp %arg7 : f64
    linalg.yield %39 : f64
  } -> tensor<{{k}}x{{d}}xf64>

  %sum_qs_init = linalg.fill(%zero, %sum_qs_space) : f64, tensor<{{k}}xf64> -> tensor<{{k}}xf64>
  %sum_qs = linalg.generic
    {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]}
    ins(%Qs : tensor<{{k}}x{{d}}xf64>)
    outs(%sum_qs_init : tensor<{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %39 = arith.addf %arg7, %arg8 : f64
    linalg.yield %39 : f64
  } -> tensor<{{k}}xf64>

  %g = arith.constant 1.0 : f64
  %dalphas_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>
  %dmeans_space = linalg.init_tensor [{{k}}, {{d}}] : tensor<{{k}}x{{d}}xf64>
  %dQdiags_space = linalg.init_tensor [{{k}}, {{d}}] : tensor<{{k}}x{{d}}xf64>
  %dsum_qs_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>


  %dmain_term_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>
  %dQs_space = linalg.init_tensor [{{k}}, {{d}}] : tensor<{{k}}x{{d}}xf64>
  %dLs_space = linalg.init_tensor [{{k}}, {{tri_size}}] : tensor<{{k}}x{{tri_size}}xf64>
  %dalphas = linalg.fill(%zero, %dalphas_space) : f64, tensor<{{k}}xf64> -> tensor<{{k}}xf64>
  %dmeans = linalg.fill(%zero, %dmeans_space) : f64, tensor<{{k}}x{{d}}xf64> -> tensor<{{k}}x{{d}}xf64>
  %dQdiags = linalg.fill(%zero, %dQdiags_space) : f64, tensor<{{k}}x{{d}}xf64> -> tensor<{{k}}x{{d}}xf64>
  %dsum_qs = linalg.fill(%zero, %dsum_qs_space) : f64, tensor<{{k}}xf64> -> tensor<{{k}}xf64>
  %dLs = linalg.fill(%zero, %dLs_space) : f64, tensor<{{k}}x{{tri_size}}xf64> -> tensor<{{k}}x{{tri_size}}xf64>
  %msqnorm_space = linalg.init_tensor [] : tensor<f64>
  %msqnorm_init = linalg.fill(%zero, %msqnorm_space) : f64, tensor<f64> -> tensor<f64>
  %se_space = linalg.init_tensor [] : tensor<f64>

  %xcentered_cache = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %Qxcentered_cache = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %ck = arith.constant {{k}} : index
  %cn = arith.constant {{n}} : index
  %cd = arith.constant {{d}} : index
  %half = arith.constant 0.5 : f64
  %res:5 = scf.for %iv = %c0 to %cn step %c1 iter_args(%dalphas_iter = %dalphas, %dsum_qs_iter = %dsum_qs, %dmeans_outer = %dmeans, %dQdiags_outer = %dQdiags, %dLs_outer = %dLs) -> (tensor<{{k}}xf64>, tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{tri_size}}xf64>) {
    %main_term = scf.for %kv = %c0 to %ck step %c1 iter_args(%mt_iter = %main_term_space) -> tensor<{{k}}xf64> {
      %x_slice = tensor.extract_slice %x[%iv, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
      %means_slice = tensor.extract_slice %means[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %xcentered = linalg.generic
        { indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"] }
        ins(%x_slice, %means_slice : tensor<{{d}}xf64>, tensor<{{d}}xf64>)
        outs(%xcentered_space : tensor<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.subf %arg0, %arg1 : f64
        linalg.yield %0 : f64
      } -> tensor<{{d}}xf64>
      // cache xcentered
      %xcentered_m = memref.buffer_cast %xcentered : memref<{{d}}xf64>
      %xview = memref.subview %xcentered_cache[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      linalg.copy(%xcentered_m, %xview) : memref<{{d}}xf64>, memref<{{d}}xf64, #view>

      %Qdiags_slice = tensor.extract_slice %Qdiags[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %Ls_slice = tensor.extract_slice %Ls[%kv, 0] [1, {{tri_size}}] [1, 1] : tensor<{{k}}x{{tri_size}}xf64> to tensor<{{tri_size}}xf64>
      %out0 = linalg.generic
        { indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"] }
        ins(%Qdiags_slice, %xcentered : tensor<{{d}}xf64>, tensor<{{d}}xf64>)
        outs(%Qxcentered_space : tensor<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        linalg.yield %0 : f64
      } -> tensor<{{d}}xf64>
      %out1 = call @Qtimesx(%Ls_slice, %xcentered) : (tensor<{{tri_size}}xf64>, tensor<{{d}}xf64>) -> tensor<{{d}}xf64>
      %Qxcentered = arith.addf %out0, %out1 : tensor<{{d}}xf64>
      // %Qxcentered = linalg.matvec ins(%Ls_slice, %xcentered : tensor<{{d}}x{{d}}xf64>, tensor<{{d}}xf64>) outs(%out0 : tensor<{{d}}xf64>) -> tensor<{{d}}xf64>

      // cache Qxcentered
      %Qxcentered_m = memref.buffer_cast %Qxcentered : memref<{{d}}xf64>
      %qview = memref.subview %Qxcentered_cache[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      linalg.copy(%Qxcentered_m, %qview) : memref<{{d}}xf64>, memref<{{d}}xf64, #view>

      %msqnorm = linalg.generic
        {
          indexing_maps = [#map2, affine_map<(d0) -> ()>],
          iterator_types = ["reduction"]
        }
        ins(%Qxcentered : tensor<{{d}}xf64>)
        outs(%msqnorm_init : tensor<f64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.mulf %arg0, %arg0 : f64
        %1 = arith.addf %0, %arg1 : f64
        linalg.yield %1 : f64
      } -> tensor<f64>
      %msqnorm_val = tensor.extract %msqnorm[] : tensor<f64>
      %hmsqnorm = arith.mulf %half, %msqnorm_val : f64
      %a_k = tensor.extract %alphas[%kv] : tensor<{{k}}xf64>
      %sum_q_k = tensor.extract %sum_qs[%kv] : tensor<{{k}}xf64>
      %mt0 = arith.addf %a_k, %sum_q_k : f64
      %mt1 = arith.subf %mt0, %hmsqnorm : f64
      %mt_next = tensor.insert %mt1 into %mt_iter[%kv] : tensor<{{k}}xf64>
      scf.yield %mt_next : tensor<{{k}}xf64>
    }

    // grad_logsumexp
    %max_init_val = tensor.extract %main_term[%c0] : tensor<{{k}}xf64>
    %max_init = tensor.insert %max_init_val into %se_space[] : tensor<f64>
    %max = linalg.generic
      { indexing_maps = [#map2, affine_map<(d0) -> ()>], iterator_types = ["reduction"] }
      ins(%main_term : tensor<{{k}}xf64>)
      outs(%max_init : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %p = arith.cmpf ogt, %arg0, %arg1 : f64
      %0 = select %p, %arg0, %arg1 : f64
      linalg.yield %0 : f64
    } -> tensor<f64>
    %max_val = tensor.extract %max[] : tensor<f64>

    %se_init = linalg.fill(%zero, %se_space) : f64, tensor<f64> -> tensor<f64>
    %sumexp = linalg.generic
      { indexing_maps = [#map2, affine_map<(d0) -> ()>], iterator_types = ["reduction"] }
      ins(%main_term : tensor<{{k}}xf64>)
      outs(%se_init : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %pre = arith.subf %arg0, %max_val : f64
      %0 = math.exp %pre : f64
      %1 = arith.addf %0, %arg1 : f64
      linalg.yield %1 : f64
    } -> tensor<f64>
    %se_val = tensor.extract %sumexp[] : tensor<f64>
    %lse = math.log %se_val : f64
    %logsumexp = arith.addf %lse, %max_val : f64
    %dmain_term = linalg.generic
      { indexing_maps = [#map2, #map2], iterator_types = ["parallel"]}
      ins(%main_term : tensor<{{k}}xf64>)
      outs(%dmain_term_space : tensor<{{k}}xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.subf %arg0, %logsumexp : f64
      %1 = math.exp %0 : f64
      %2 = arith.mulf %1, %g : f64
      linalg.yield %2 : f64
    } -> tensor<{{k}}xf64>
    // end grad_logsumexp

    %dalphas_next = linalg.generic
      { indexing_maps = [#map2, #map2], iterator_types = ["parallel"] }
      ins(%dmain_term : tensor<{{k}}xf64>)
      outs(%dalphas_iter : tensor<{{k}}xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.addf %arg0, %arg1 : f64
      linalg.yield %0 : f64
    } -> tensor<{{k}}xf64>
    %dsum_qs_next = linalg.generic
      { indexing_maps = [#map2, #map2], iterator_types = ["parallel"] }
      ins(%dmain_term : tensor<{{k}}xf64>)
      outs(%dsum_qs_iter : tensor<{{k}}xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.addf %arg0, %arg1 : f64
      linalg.yield %0 : f64
    } -> tensor<{{k}}xf64>

    %inner_next:3 = scf.for %kv = %c0 to %ck step %c1 iter_args(%dmeans_iter = %dmeans_outer, %dQdiags_iter = %dQdiags_outer, %dLs_iter = %dLs_outer) -> (tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{tri_size}}xf64>) {
      // Read primal values from cache
      %xcentered_view = memref.subview %xcentered_cache[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      %xcentered_casted = memref.cast %xcentered_view : memref<{{d}}xf64, #view> to memref<{{d}}xf64>
      %xcentered = memref.tensor_load %xcentered_casted : memref<{{d}}xf64>
      %Qxcentered_view = memref.subview %Qxcentered_cache[%kv, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #view>
      %Qxcentered_casted = memref.cast %Qxcentered_view : memref<{{d}}xf64, #view> to memref<{{d}}xf64>
      %Qxcentered = memref.tensor_load %Qxcentered_casted : memref<{{d}}xf64>

      %dmsqnorm_0 = tensor.extract %dmain_term[%kv] : tensor<{{k}}xf64>
      %dmsqnorm = arith.negf %dmsqnorm_0 : f64
      %dQxcentered_space = linalg.init_tensor [{{d}}] : tensor<{{d}}xf64>
      %dQxcentered = linalg.generic
        { indexing_maps = [#map2, #map2], iterator_types = ["parallel"] }
        ins(%Qxcentered : tensor<{{d}}xf64>)
        outs(%dQxcentered_space : tensor<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.mulf %arg0, %dmsqnorm : f64
        linalg.yield %0 : f64
      } -> tensor<{{d}}xf64>
      %Qdiags_slice = tensor.extract_slice %Qdiags[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %Ls_slice = tensor.extract_slice %Ls[%kv, 0] [1, {{tri_size}}] [1, 1] : tensor<{{k}}x{{tri_size}}xf64> to tensor<{{tri_size}}xf64>
      %dxcentered_0 = arith.mulf %Qdiags_slice, %dQxcentered : tensor<{{d}}xf64>
      // %dxcentered = call @vecmat(%dQxcentered, %Ls_slice, %dxcentered_0) : (tensor<{{d}}xf64>, tensor<{{tri_size}}xf64>, tensor<{{d}}xf64>) -> tensor<{{d}}xf64>

      %dLs_slice = tensor.extract_slice %dLs_iter[%kv, 0] [1, {{tri_size}}] [1, 1] : tensor<{{k}}x{{tri_size}}xf64> to tensor<{{tri_size}}xf64>
      // %dLs_slice_next = call @outer_product(%dQxcentered, %xcentered, %dLs_slice) : (tensor<{{d}}xf64>, tensor<{{d}}xf64>, tensor<{{tri_size}}xf64>) -> tensor<{{tri_size}}xf64>

      %Qtimesxadj:2 = scf.for %mv = %c0 to %cd step %c1 iter_args(%outer = %dxcentered_0, %outer_Lsb = %dLs_slice) -> (tensor<{{d}}xf64>, tensor<{{tri_size}}xf64>) {
        %Lidx_0 = arith.muli %c2, %cd : index
        %Lidx_1 = arith.subi %Lidx_0, %mv : index
        %Lidx_2 = arith.subi %Lidx_1, %c1 : index
        %Lidx_3 = arith.muli %Lidx_2, %mv : index
        %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

        %iv_plus_1 = arith.addi %mv, %c1 : index
        %outer_next:2 = scf.for %jv = %iv_plus_1 to %cd step %c1 iter_args(%out_iter = %outer, %dL_slice = %outer_Lsb) -> (tensor<{{d}}xf64>, tensor<{{tri_size}}xf64>) {
          %Lidx_5 = arith.addi %Lidx_4, %jv : index
          %Lidx = arith.subi %Lidx_5, %iv_plus_1 : index
          %0 = tensor.extract %Ls_slice[%Lidx] : tensor<{{tri_size}}xf64>
          %1 = tensor.extract %dQxcentered[%jv] : tensor<{{d}}xf64>
          %2 = tensor.extract %out_iter[%mv] : tensor<{{d}}xf64>
          %3 = arith.mulf %0, %1 : f64
          %4 = arith.addf %3, %2 : f64
          %out_next = tensor.insert %4 into %out_iter[%mv] : tensor<{{d}}xf64>

          %xval = tensor.extract %xcentered[%mv] : tensor<{{d}}xf64>
          %dLval = tensor.extract %dL_slice[%Lidx] : tensor<{{tri_size}}xf64>
          %5 = arith.mulf %xval, %1 : f64
          %6 = arith.addf %5, %dLval : f64
          %dL_next = tensor.insert %6 into %dL_slice[%Lidx] : tensor<{{tri_size}}xf64>
          scf.yield %out_next, %dL_next : tensor<{{d}}xf64>, tensor<{{tri_size}}xf64>
        }
        scf.yield %outer_next#0, %outer_next#1 : tensor<{{d}}xf64>, tensor<{{tri_size}}xf64>
      }
      // %dLs_next = tensor.insert_slice %dLs_slice_next into %dLs_iter[%kv, 0] [1, {{tri_size}}] [1, 1] : tensor<{{tri_size}}xf64> into tensor<{{k}}x{{tri_size}}xf64>
      %dLs_next = tensor.insert_slice %Qtimesxadj#1 into %dLs_iter[%kv, 0] [1, {{tri_size}}] [1, 1] : tensor<{{tri_size}}xf64> into tensor<{{k}}x{{tri_size}}xf64>

      %dmeans_slice = tensor.extract_slice %dmeans_iter[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %dmeans_slice_next = linalg.generic
        { indexing_maps = [#map2, #map2], iterator_types = ["parallel"] }
        // ins(%dxcentered : tensor<{{d}}xf64>)
        ins(%Qtimesxadj#0 : tensor<{{d}}xf64>)
        outs(%dmeans_slice : tensor<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %0 = arith.subf %arg1, %arg0 : f64
        linalg.yield %0 : f64
      } -> tensor<{{d}}xf64>
      %dmeans_next = tensor.insert_slice %dmeans_slice_next into %dmeans_iter[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{d}}xf64> into tensor<{{k}}x{{d}}xf64>

      %dQdiags_slice = tensor.extract_slice %dQdiags_iter[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %dQdiags_slice_next = linalg.generic
        { indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"] }
        ins(%dQxcentered, %xcentered : tensor<{{d}}xf64>, tensor<{{d}}xf64>)
        outs(%dQdiags_slice : tensor<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        %1 = arith.addf %0, %arg2 : f64
        linalg.yield %1 : f64
      } -> tensor<{{d}}xf64>
      %dQdiags_next = tensor.insert_slice %dQdiags_slice_next into %dQdiags_iter[%kv, 0] [1, {{d}}] [1, 1] : tensor<{{d}}xf64> into tensor<{{k}}x{{d}}xf64>

      scf.yield %dmeans_next, %dQdiags_next, %dLs_next : tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{tri_size}}xf64>
    }
    scf.yield %dalphas_next, %dsum_qs_next, %inner_next#0, %inner_next#1, %inner_next#2 : tensor<{{k}}xf64>, tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{tri_size}}xf64>
  }

  %dQs = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>, #map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%res#1, %res#3, %Qdiags : tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>)
    outs(%dQs_space : tensor<{{k}}x{{d}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg1, %arg2 : f64
    %1 = arith.addf %0, %arg0 : f64
    linalg.yield %1 : f64
  } -> tensor<{{k}}x{{d}}xf64>

  return %res#0, %res#2, %dQs, %res#4 : tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{tri_size}}xf64>
}

//
// LAGrad Main Term
//
func @mmain_term(
  %alphas: tensor<{{k}}xf64>,
  %means: tensor<{{k}}x{{d}}xf64>,
  %Qs: tensor<{{k}}x{{d}}xf64>,
  %Ls: tensor<{{k}}x{{d}}x{{d}}xf64, "pltri">,
  %x: tensor<{{n}}x{{d}}xf64>
) -> f64 {
  %zero = arith.constant 0.0 : f64
  %Qdiags_space = linalg.init_tensor [{{k}}, {{d}}] : tensor<{{k}}x{{d}}xf64>
  %sum_qs_space = arith.constant dense<0.0> : tensor<{{k}}xf64>
  %len_d_zero = arith.constant dense<0.0> : tensor<{{d}}xf64>
  %main_term_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>
  %out1_space = linalg.init_tensor [{{d}}] : tensor<{{d}}xf64>
  %out1_init = linalg.fill(%zero, %out1_space) : f64, tensor<{{d}}xf64> -> tensor<{{d}}xf64>
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
      %xcentered = arith.subf %x_slice, %means_slice {lagrad_should_cache} : tensor<{{d}}xf64>

      %Qdiags_slice = tensor.extract_slice %Qdiags[%ik, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %Ltri_slice = tensor.extract_slice %Ls[%ik, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : tensor<{{k}}x{{d}}x{{d}}xf64, "pltri"> to tensor<{{d}}x{{d}}xf64, "pltri">

      // inlined Qtimesx
      // Elementwise multiplication
      %out_0 = arith.mulf %Qdiags_slice, %xcentered : tensor<{{d}}xf64>

      // The triangular matrix-vector multiplication
      %out_1 = linalg.generic
        {
          doc = "Column major TRMV (for packing)",
          indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>],
          iterator_types = ["reduction", "parallel"]
        }
        ins(%Ltri_slice, %xcentered : tensor<{{d}}x{{d}}xf64, "pltri">, tensor<{{d}}xf64>) outs(%out1_init : tensor<{{d}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        %1 = arith.addf %0, %arg2 : f64
        linalg.yield %1 : f64
      } -> tensor<{{d}}xf64>
      // %out_1 = linalg.matvec ins(%Ltri_slice, %xcentered : tensor<{{d}}x{{d}}xf64>, tensor<{{d}}xf64>) outs(%len_d_zero : tensor<{{d}}xf64>) -> tensor<{{d}}xf64>
      %Qxcentered = arith.addf %out_0, %out_1 {lagrad_should_cache} : tensor<{{d}}xf64>

      %msqnorm_t = linalg.generic {indexing_maps = [#map2, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%Qxcentered : tensor<{{d}}xf64>) outs(%zerod_tensor : tensor<f64>) {
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
    } {lagrad_should_cache}

    // logsumexp %main_term inlined
    // find the max
    %max_init_val = tensor.extract %main_term[%c0] : tensor<{{k}}xf64>
    %max_init = tensor.insert %max_init_val into %max_space[] : tensor<f64>

    %max_t = linalg.generic
      {indexing_maps = [#map2, affine_map<(d0) -> ()>], iterator_types = ["reduction"]}
      ins(%main_term : tensor<{{k}}xf64>)
      outs(%max_init : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %p = arith.cmpf "ogt", %arg0, %arg1 : f64
      %next = select %p, %arg0, %arg1 : f64
      linalg.yield %next : f64
    } -> tensor<f64>

    %max = tensor.extract %max_t[] : tensor<f64>
    %se_noadd_t = linalg.generic
      {indexing_maps = [#map2, affine_map<(d0) -> ()>], iterator_types = ["reduction"]}
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
    %slse_next = arith.addf %slse_iv, %lse : f64
    scf.yield %slse_next : f64
  }
  return %slse : f64
}

func @lagrad_main_term(
  %alphas: tensor<{{k}}xf64>,
  %means: tensor<{{k}}x{{d}}xf64>,
  %Qs: tensor<{{k}}x{{d}}xf64>,
  %Ls: tensor<{{k}}x{{d}}x{{d}}xf64, "pltri">,
  %x: tensor<{{n}}x{{d}}xf64>
) -> (tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64, "pltri">) {
  %f = constant @mmain_term : (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64, "pltri">,
    tensor<{{n}}x{{d}}xf64>
  ) -> f64
  %df = standalone.grad %f {of = [0, 1, 2, 3]} : (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64, "pltri">,
    tensor<{{n}}x{{d}}xf64>
  ) -> f64, (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64, "pltri">,
    tensor<{{n}}x{{d}}xf64>
  ) -> (tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64, "pltri">)
  %res:4 = call_indirect %df(%alphas, %means, %Qs, %Ls, %x) : (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64, "pltri">,
    tensor<{{n}}x{{d}}xf64>
  ) -> (tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64, "pltri">)
  return %res#0, %res#1, %res#2, %res#3 : tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64, "pltri">
}
