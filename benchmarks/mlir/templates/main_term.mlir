#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map9 = affine_map<(d0) -> (d0)>
#map11 = affine_map<(d0) -> ()>

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main_term(
  %alphas: tensor<{{k}}xf64>,
  %means: tensor<{{k}}x{{d}}xf64>,
  %Qs: tensor<{{k}}x{{d}}xf64>,
  %Ls: tensor<{{k}}x{{d}}x{{d}}xf64>,
  %x: tensor<{{n}}x{{d}}xf64>
) -> f64 {
  %zero = arith.constant 0.0 : f64
  %Qdiags_space = linalg.init_tensor [{{k}}, {{d}}] : tensor<{{k}}x{{d}}xf64>
  %sum_qs_space = arith.constant dense<0.0> : tensor<{{k}}xf64>
  %len_d_zero = arith.constant dense<0.0> : tensor<{{d}}xf64>
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
      %xcentered = arith.subf %x_slice, %means_slice : tensor<{{d}}xf64>

      %Qdiags_slice = tensor.extract_slice %Qdiags[%ik, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %Ltri_slice = tensor.extract_slice %Ls[%ik, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : tensor<{{k}}x{{d}}x{{d}}xf64> to tensor<{{d}}x{{d}}xf64>

      // inlined Qtimesx
      // Elementwise multiplication
      %out_0 = arith.mulf %Qdiags_slice, %xcentered : tensor<{{d}}xf64>

      // The triangular matrix-vector multiplication
      %out_1 = linalg.matvec ins(%Ltri_slice, %xcentered : tensor<{{d}}x{{d}}xf64>, tensor<{{d}}xf64>) outs(%len_d_zero : tensor<{{d}}xf64>) -> tensor<{{d}}xf64>
      %Qxcentered = arith.addf %out_0, %out_1 : tensor<{{d}}xf64>

      // %reduced = linalg.generic
      //   {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]}
      //   ins(%out_1 : tensor<{{d}}xf64>)
      //   outs(%zerod_tensor : tensor<f64>) {
      // ^bb0(%arg0: f64, %arg1: f64):
      //   %0 = arith.addf %arg0, %arg1 : f64
      //   linalg.yield %0 : f64
      // } -> tensor<f64>
      // %r0 = tensor.extract %reduced[] : tensor<f64>
      // %main_term_ik = arith.addf %mt1, %r0 : f64
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

    // %lse_t = linalg.generic
    //   {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
    //   ins(%main_term : tensor<{{k}}xf64>)
    //   outs(%zerod_tensor : tensor<f64>) {
    // ^bb0(%arg0: f64, %arg1: f64):
    //   %0 = arith.addf %arg0, %arg1 : f64
    //   linalg.yield %0 : f64
    // } -> tensor<f64>
    // %lse = tensor.extract %lse_t[] : tensor<f64>

    // logsumexp %main_term inlined
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
  return %slse : f64
}

// func @main_term_hand_mlir(
//   %alphas: tensor<{{k}}xf64>,
//   %means: tensor<{{k}}x{{d}}xf64>,
//   %Qs: tensor<{{k}}x{{d}}xf64>,
//   %Ls: tensor<{{k}}x{{d}}x{{d}}xf64>,
//   %x: tensor<{{n}}x{{d}}xf64>
// ) -> tensor<{{k}}xf64> {
//   %dslse = arith.constant 1.0 : f64
//   %one = arith.constant 1.0 : f64
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %cn = arith.constant {{n}} : index
//   %main_term_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>
//   scf.for %iv = %c0 to %cn step %c1 {
//     %main_term = scf.for %ik = %c0 to %ck step %c1 iter_args(%mt_iter = %main_term_space) -> tensor<{{k}}xf64> {
//       // Subtract
//       %x_slice = tensor.extract_slice %x[%ix, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
//       %means_slice = tensor.extract_slice %means[%ik, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
//       %xcentered = arith.subf %x_slice, %means_slice : tensor<{{d}}xf64>

//       %Qdiags_slice = tensor.extract_slice %Qdiags[%ik, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
//       %Ltri_slice = tensor.extract_slice %Ls[%ik, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : tensor<{{k}}x{{d}}x{{d}}xf64> to tensor<{{d}}x{{d}}xf64>

//       // inlined Qtimesx
//       // Elementwise multiplication
//       %out_0 = arith.mulf %Qdiags_slice, %xcentered : tensor<{{d}}xf64>

//       // The triangular matrix-vector multiplication
//       %out_1 = linalg.matvec ins(%Ltri_slice, %xcentered : tensor<{{d}}x{{d}}xf64>, tensor<{{d}}xf64>) outs(%len_d_zero : tensor<{{d}}xf64>) -> tensor<{{d}}xf64>
//       %Qxcentered = arith.addf %out_0, %out_1 : tensor<{{d}}xf64>

//       // %reduced = linalg.generic
//       //   {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]}
//       //   ins(%out_1 : tensor<{{d}}xf64>)
//       //   outs(%zerod_tensor : tensor<f64>) {
//       // ^bb0(%arg0: f64, %arg1: f64):
//       //   %0 = arith.addf %arg0, %arg1 : f64
//       //   linalg.yield %0 : f64
//       // } -> tensor<f64>
//       // %r0 = tensor.extract %reduced[] : tensor<f64>
//       // %main_term_ik = arith.addf %mt1, %r0 : f64
//       %msqnorm_t = linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%Qxcentered : tensor<{{d}}xf64>) outs(%zerod_tensor : tensor<f64>) {
//       ^bb0(%arg0: f64, %arg1: f64):
//         %0 = arith.mulf %arg0, %arg0 : f64
//         %1 = arith.addf %0, %arg1 : f64
//         linalg.yield %1 : f64
//       } -> tensor<f64>
//       %msqnorm = tensor.extract %msqnorm_t[] : tensor<f64>
//       %hmsqnorm = arith.mulf %msqnorm, %half : f64
//       %a_ik = tensor.extract %alphas[%ik] : tensor<{{k}}xf64>
//       %q_ik = tensor.extract %sum_qs[%ik] : tensor<{{k}}xf64>
//       %sum_aq = arith.addf %a_ik, %q_ik : f64
//       %main_term_ik = arith.subf %sum_aq, %hmsqnorm : f64
//       %main_term_next = tensor.insert %main_term_ik into %mt_iter[%ik] : tensor<{{k}}xf64>
//       scf.yield %main_term_next : tensor<{{k}}xf64>
//     }
//     %dlse = arith.constant 1.0 : f64

//     // primal values
//     %max = tensor.extract %max_t[] : tensor<f64>
//     %se_noadd_t = linalg.generic
//       {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
//       ins(%main_term : tensor<{{k}}xf64>)
//       outs(%zerod_tensor : tensor<f64>) {
//     ^bb0(%arg0: f64, %arg1: f64):
//       %0 = arith.subf %arg0, %max : f64
//       %1 = math.exp %0 : f64
//       %2 = arith.addf %1, %arg1 : f64
//       linalg.yield %2 : f64
//     } -> tensor<f64>
//     %se_noadd = tensor.extract %se_noadd_t[] : tensor<f64>
//     // end primal

//     %dlse_noadd = arith.divf %dlse, %se_noadd : f64
//   }
//   return
// }

func @main_term_lagrad(
  %alphas: tensor<{{k}}xf64>,
  %means: tensor<{{k}}x{{d}}xf64>,
  %Qs: tensor<{{k}}x{{d}}xf64>,
  %Ls: tensor<{{k}}x{{d}}x{{d}}xf64>,
  %x: tensor<{{n}}x{{d}}xf64>
) -> (
  tensor<{{k}}xf64>,
  tensor<{{k}}x{{d}}xf64>,
  tensor<{{k}}x{{d}}xf64>,
  tensor<{{k}}x{{d}}x{{d}}xf64>
) {
  %f = constant @main_term : (tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>, tensor<{{n}}x{{d}}xf64>) -> f64
  %df = standalone.grad %f {of = [0, 1, 2, 3]} : (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64>,
    tensor<{{n}}x{{d}}xf64>
  ) -> f64, (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64>,
    tensor<{{n}}x{{d}}xf64>    
  ) -> (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64>
  )
  %res:4 = call_indirect %df(%alphas, %means, %Qs, %Ls, %x) : (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64>,
    tensor<{{n}}x{{d}}xf64>    
  ) -> (
    tensor<{{k}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}xf64>,
    tensor<{{k}}x{{d}}x{{d}}xf64>
  )
  return %res#0, %res#1, %res#2, %res#3 : tensor<{{k}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>, tensor<{{k}}x{{d}}x{{d}}xf64>
}
