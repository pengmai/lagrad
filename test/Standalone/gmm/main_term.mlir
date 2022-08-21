#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map9 = affine_map<(d0) -> (d0)>
#map11 = affine_map<(d0) -> ()>

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func private @main_term(
  %alphas: tensor<4xf64>,
  %means: tensor<4x2xf64>,
  %Qs: tensor<4x2xf64>,
  %Ls: tensor<4x2x2xf64>,
  %x: tensor<10x2xf64>
) -> f64 {
  %zero = arith.constant 0.0 : f64
  %Qdiags_space = linalg.init_tensor [4, 2] : tensor<4x2xf64>
  %sum_qs_space = arith.constant dense<0.0> : tensor<4xf64>
  %len_d_zero = arith.constant dense<0.0> : tensor<2xf64>
  %main_term_space = linalg.init_tensor [4] : tensor<4xf64>
  %zerod_tensor = arith.constant dense<0.0> : tensor<f64>
  %max_space = linalg.init_tensor [] : tensor<f64>

  // This is the preprocess Qs implementation in the original function.
  %Qdiags = linalg.generic
    {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
    ins(%Qs : tensor<4x2xf64>)
    outs(%Qdiags_space : tensor<4x2xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %39 = math.exp %arg7 : f64
    linalg.yield %39 : f64
  } -> tensor<4x2xf64>

  %sum_qs = linalg.generic
    {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]}
    ins(%Qs : tensor<4x2xf64>)
    outs(%sum_qs_space : tensor<4xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %39 = arith.addf %arg7, %arg8 : f64
    linalg.yield %39 : f64
  } -> tensor<4xf64>

  %half = arith.constant 0.5 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cn = arith.constant 10 : index
  %ck = arith.constant 4 : index
  %cd = arith.constant 2 : index
  %slse = scf.for %ix = %c0 to %cn step %c1 iter_args(%slse_iv = %zero) -> f64 {
    %main_term = scf.for %ik = %c0 to %ck step %c1 iter_args(%mt_iter = %main_term_space) -> tensor<4xf64> {
      // Subtract
      %x_slice = tensor.extract_slice %x[%ix, 0] [1, 2] [1, 1] : tensor<10x2xf64> to tensor<2xf64>
      %means_slice = tensor.extract_slice %means[%ik, 0] [1, 2] [1, 1] : tensor<4x2xf64> to tensor<2xf64>
      %xcentered = arith.subf %x_slice, %means_slice : tensor<2xf64>

      %Qdiags_slice = tensor.extract_slice %Qdiags[%ik, 0] [1, 2] [1, 1] : tensor<4x2xf64> to tensor<2xf64>
      %Ltri_slice = tensor.extract_slice %Ls[%ik, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<4x2x2xf64> to tensor<2x2xf64>

      // inlined Qtimesx
      // Elementwise multiplication
      %Qxcentered_0 = arith.mulf %Qdiags_slice, %xcentered : tensor<2xf64>

      // The triangular matrix-vector multiplication
      %matvec_1 = linalg.matvec {library_call = "dmatvec"} ins(%Ltri_slice, %xcentered : tensor<2x2xf64>, tensor<2xf64>) outs(%len_d_zero : tensor<2xf64>) -> tensor<2xf64>
      %Qxcentered = arith.addf %Qxcentered_0, %matvec_1 : tensor<2xf64>
      %msqnorm_0 = arith.mulf %Qxcentered, %Qxcentered : tensor<2xf64>
      %msqnorm_t = linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%msqnorm_0 : tensor<2xf64>) outs(%zerod_tensor : tensor<f64>) {
      ^bb0(%arg0: f64, %arg1: f64):
        %1 = arith.addf %arg0, %arg1 : f64
        linalg.yield %1 : f64
      } -> tensor<f64>
      %msqnorm = tensor.extract %msqnorm_t[] : tensor<f64>
      %hmsqnorm = arith.mulf %msqnorm, %half : f64
      %a_ik = tensor.extract %alphas[%ik] : tensor<4xf64>
      %q_ik = tensor.extract %sum_qs[%ik] : tensor<4xf64>
      %sum_aq = arith.addf %a_ik, %q_ik : f64
      %main_term_ik = arith.subf %sum_aq, %hmsqnorm : f64
      %main_term_next = tensor.insert %main_term_ik into %mt_iter[%ik] : tensor<4xf64>
      scf.yield %main_term_next : tensor<4xf64>
    }

    // logsumexp %main_term inlined
    // find the max
    %max_init_val = tensor.extract %main_term[%c0] : tensor<4xf64>
    %max_init = tensor.insert %max_init_val into %max_space[] : tensor<f64>

    %max_t = linalg.generic
      {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
      ins(%main_term : tensor<4xf64>)
      outs(%max_init : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %p = arith.cmpf "ogt", %arg0, %arg1 : f64
      %next = select %p, %arg0, %arg1 : f64
      linalg.yield %next : f64
    } -> tensor<f64>

    %max = tensor.extract %max_t[] : tensor<f64>
    %se_noadd_t = linalg.generic
      {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
      ins(%main_term : tensor<4xf64>)
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

func @main() {
  %alphas = arith.constant dense<[0.0000, 0.1315, 0.7556, 0.4587]> : tensor<4xf64>
  %means = arith.constant dense<[
    [0.5328, 0.2190], [0.0470, 0.6789],
    [0.6793, 0.9347], [0.3835, 0.5194]
  ]> : tensor<4x2xf64>
  %Qs = arith.constant dense<[
    [0.8310, 0.0346], [0.0535, 0.5297],
    [0.6711, 0.0077], [0.3834, 0.0668]
  ]> : tensor<4x2xf64>
  %Ls = arith.constant dense<[
    [[0.4175, 0.6868], [0.5890, 0.9304]],
    [[0.8462, 0.5269], [0.0920, 0.6539]],
    [[0.4160, 0.7012], [0.9103, 0.7622]],
    [[0.2625, 0.0475], [0.7361, 0.3282]]
  ]> : tensor<4x2x2xf64>
  %x = arith.constant dense<[
    [0.6326, 0.7564],
    [0.9910, 0.3653],
    [0.2470, 0.9826],
    [0.7227, 0.7534],
    [0.6515, 0.0727],
    [0.6316, 0.8847],
    [0.2727, 0.4364],
    [0.7665, 0.4777],
    [0.2378, 0.2749],
    [0.3593, 0.1665]
  ]> : tensor<10x2xf64>

  %f = constant @main_term : (tensor<4xf64>, tensor<4x2xf64>, tensor<4x2xf64>, tensor<4x2x2xf64>, tensor<10x2xf64>) -> f64
  %df = standalone.grad %f {of = [0, 1, 2, 3]} : (
    tensor<4xf64>,
    tensor<4x2xf64>,
    tensor<4x2xf64>,
    tensor<4x2x2xf64>,
    tensor<10x2xf64>
  ) -> f64, (
    tensor<4xf64>,
    tensor<4x2xf64>,
    tensor<4x2xf64>,
    tensor<4x2x2xf64>,
    tensor<10x2xf64>    
  ) -> (
    tensor<4xf64>,
    tensor<4x2xf64>,
    tensor<4x2xf64>,
    tensor<4x2x2xf64>
  )
  %res:4 = call_indirect %df(%alphas, %means, %Qs, %Ls, %x) : (
    tensor<4xf64>,
    tensor<4x2xf64>,
    tensor<4x2xf64>,
    tensor<4x2x2xf64>,
    tensor<10x2xf64>    
  ) -> (
    tensor<4xf64>,
    tensor<4x2xf64>,
    tensor<4x2xf64>,
    tensor<4x2x2xf64>
  )

  %U0 = tensor.cast %res#0 : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U0) : (tensor<*xf64>) -> ()
  %U1 = tensor.cast %res#1 : tensor<4x2xf64> to tensor<*xf64>
  call @print_memref_f64(%U1) : (tensor<*xf64>) -> ()
  %U2 = tensor.cast %res#2 : tensor<4x2xf64> to tensor<*xf64>
  call @print_memref_f64(%U2) : (tensor<*xf64>) -> ()
  %U3 = tensor.cast %res#3 : tensor<4x2x2xf64> to tensor<*xf64>
  call @print_memref_f64(%U3) : (tensor<*xf64>) -> ()
  return
}
