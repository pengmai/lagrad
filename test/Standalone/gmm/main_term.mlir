// #map0 = affine_map<(d0, d1) -> (d0, d1)>
// #map1 = affine_map<(d0, d1) -> (d0)>
// #map9 = affine_map<(d0) -> (d0)>
// #map11 = affine_map<(d0) -> ()>

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

// func @main_term(
//   %alphas: tensor<4xf64>,
//   %means: tensor<4x2xf64>,
//   %Qs: tensor<4x2xf64>,
//   %Ls: tensor<4x2x2xf64>,
//   %x: tensor<10x2xf64>
// ) -> f64 {
//   %zero = arith.constant 0.0 : f64
//   %Qdiags_space = linalg.init_tensor [4, 2] : tensor<4x2xf64>
//   %sum_qs_space = arith.constant dense<0.0> : tensor<4xf64>
//   %len_d_zero = arith.constant dense<0.0> : tensor<2xf64>
//   %main_term_space = linalg.init_tensor [4] : tensor<4xf64>
//   %zerod_tensor = arith.constant dense<0.0> : tensor<f64>
//   %max_space = linalg.init_tensor [] : tensor<f64>

//   // This is the preprocess Qs implementation in the original function.
//   %Qdiags = linalg.generic
//     {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
//     ins(%Qs : tensor<4x2xf64>)
//     outs(%Qdiags_space : tensor<4x2xf64>) {
//   ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
//     %39 = math.exp %arg7 : f64
//     linalg.yield %39 : f64
//   } -> tensor<4x2xf64>

//   %sum_qs = linalg.generic
//     {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]}
//     ins(%Qs : tensor<4x2xf64>)
//     outs(%sum_qs_space : tensor<4xf64>) {
//   ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
//     %39 = arith.addf %arg7, %arg8 : f64
//     linalg.yield %39 : f64
//   } -> tensor<4xf64>

//   %half = arith.constant 0.5 : f64
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//   %cn = arith.constant 10 : index
//   %ck = arith.constant 4 : index
//   %cd = arith.constant 2 : index
//   %slse = scf.for %ix = %c0 to %cn step %c1 iter_args(%slse_iv = %zero) -> f64 {
//     %main_term = scf.for %ik = %c0 to %ck step %c1 iter_args(%mt_iter = %main_term_space) -> tensor<4xf64> {
//       // Subtract
//       %x_slice = tensor.extract_slice %x[%ix, 0] [1, 2] [1, 1] : tensor<10x2xf64> to tensor<2xf64>
//       %means_slice = tensor.extract_slice %means[%ik, 0] [1, 2] [1, 1] : tensor<4x2xf64> to tensor<2xf64>
//       %xcentered = arith.subf %x_slice, %means_slice : tensor<2xf64>

//       %Qdiags_slice = tensor.extract_slice %Qdiags[%ik, 0] [1, 2] [1, 1] : tensor<4x2xf64> to tensor<2xf64>
//       %Ltri_slice = tensor.extract_slice %Ls[%ik, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<4x2x2xf64> to tensor<2x2xf64>

//       // inlined Qtimesx
//       // Elementwise multiplication
//       %out_0 = arith.mulf %Qdiags_slice, %xcentered : tensor<2xf64>

//       // The triangular matrix-vector multiplication
//       %out_1 = linalg.matvec ins(%Ltri_slice, %xcentered : tensor<2x2xf64>, tensor<2xf64>) outs(%len_d_zero : tensor<2xf64>) -> tensor<2xf64>
//       %Qxcentered = arith.addf %out_0, %out_1 : tensor<2xf64>

//       %msqnorm_t = linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%Qxcentered : tensor<2xf64>) outs(%zerod_tensor : tensor<f64>) {
//       ^bb0(%arg0: f64, %arg1: f64):
//         %0 = arith.mulf %arg0, %arg0 : f64
//         %1 = arith.addf %0, %arg1 : f64
//         linalg.yield %1 : f64
//       } -> tensor<f64>
//       %msqnorm = tensor.extract %msqnorm_t[] : tensor<f64>
//       %hmsqnorm = arith.mulf %msqnorm, %half : f64
//       %a_ik = tensor.extract %alphas[%ik] : tensor<4xf64>
//       %q_ik = tensor.extract %sum_qs[%ik] : tensor<4xf64>
//       %sum_aq = arith.addf %a_ik, %q_ik : f64
//       %main_term_ik = arith.subf %sum_aq, %hmsqnorm : f64
//       %main_term_next = tensor.insert %main_term_ik into %mt_iter[%ik] : tensor<4xf64>
//       scf.yield %main_term_next : tensor<4xf64>
//     }

//     // logsumexp %main_term inlined
//     // find the max
//     %max_init_val = tensor.extract %main_term[%c0] : tensor<4xf64>
//     %max_init = tensor.insert %max_init_val into %max_space[] : tensor<f64>

//     %max_t = linalg.generic
//       {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
//       ins(%main_term : tensor<4xf64>)
//       outs(%max_init : tensor<f64>) {
//     ^bb0(%arg0: f64, %arg1: f64):
//       %p = arith.cmpf "ogt", %arg0, %arg1 : f64
//       %next = scf.if %p -> (f64) {
//         scf.yield %arg0 : f64
//       } else {
//         scf.yield %arg1 : f64
//       }
//       linalg.yield %next : f64
//     } -> tensor<f64>

//     %max = tensor.extract %max_t[] : tensor<f64>
//     %se_noadd_t = linalg.generic
//       {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]}
//       ins(%main_term : tensor<4xf64>)
//       outs(%zerod_tensor : tensor<f64>) {
//     ^bb0(%arg0: f64, %arg1: f64):
//       %0 = arith.subf %arg0, %max : f64
//       %1 = math.exp %0 : f64
//       %2 = arith.addf %1, %arg1 : f64
//       linalg.yield %2 : f64
//     } -> tensor<f64>
//     %se_noadd = tensor.extract %se_noadd_t[] : tensor<f64>
//     %lse_noadd = math.log %se_noadd : f64
//     %lse = arith.addf %lse_noadd, %max : f64
//     %slse_next = arith.addf %slse_iv, %lse : f64
//     scf.yield %slse_next : f64
//   }
//   return %slse : f64
// }

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>
#map4 = affine_map<(d0, d1) -> (d1)>

func @__grad_main_term(
  %arg0: tensor<4xf64>,
  %arg1: tensor<4x2xf64>,
  %arg2: tensor<4x2xf64>,
  %arg3: tensor<4x2x2xf64>,
  %arg4: tensor<10x2xf64>,
  %darg0: tensor<4xf64>//,
  // %darg1: tensor<4x2xf64>,
  // %darg2: tensor<4x2xf64>,
  // %darg3: tensor<4x2x2xf64>
) -> (
  tensor<4xf64>//,
  // tensor<4x2xf64>,
  // tensor<4x2xf64>,
  // tensor<4x2x2xf64>
) {
  %cst = arith.constant dense<0.000000e+00> : tensor<4x2xf64>
  %c3 = arith.constant 3 : index
  %c9 = arith.constant 9 : index
  %cst_0 = arith.constant 0.000000e+00 : f64
  %cst_1 = arith.constant 1.000000e+00 : f64
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst_2 = arith.constant 5.000000e-01 : f64
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<f64>
  %cst_4 = arith.constant dense<0.000000e+00> : tensor<2xf64>
  %cst_5 = arith.constant dense<0.000000e+00> : tensor<4xf64>
  %0 = linalg.init_tensor [4, 2] : tensor<4x2xf64>
  %1 = linalg.init_tensor [4] : tensor<4xf64>
  %2 = linalg.init_tensor [] : tensor<f64>
  %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<4x2xf64>) outs(%0 : tensor<4x2xf64>) {
  ^bb0(%arg5: f64, %arg6: f64):
    %21 = math.exp %arg5 : f64
    linalg.yield %21 : f64
  } -> tensor<4x2xf64>
  %4 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<4x2xf64>) outs(%cst_5 : tensor<4xf64>) {
  ^bb0(%arg5: f64, %arg6: f64):
    %21 = arith.addf %arg5, %arg6 : f64
    linalg.yield %21 : f64
  } -> tensor<4xf64>
  %5 = linalg.init_tensor [4, 2, 2] : tensor<4x2x2xf64>
  %6 = linalg.fill(%cst_0, %5) : f64, tensor<4x2x2xf64> -> tensor<4x2x2xf64> 
  %7 = linalg.init_tensor [10, 2] : tensor<10x2xf64>
  %8 = linalg.fill(%cst_0, %7) : f64, tensor<10x2xf64> -> tensor<10x2xf64> 
  %9 = linalg.init_tensor [4, 2] : tensor<4x2xf64>
  %10 = linalg.fill(%cst_0, %9) : f64, tensor<4x2xf64> -> tensor<4x2xf64> 
  %11 = linalg.init_tensor [4, 2] : tensor<4x2xf64>
  %12 = linalg.fill(%cst_0, %11) : f64, tensor<4x2xf64> -> tensor<4x2xf64> 
  %13 = linalg.init_tensor [4] : tensor<4xf64>
  %14 = linalg.fill(%cst_0, %13) : f64, tensor<4xf64> -> tensor<4xf64>
  %17:6 = scf.for %arg5 = %c0 to %c10 step %c1 iter_args(%arg6 = %6, %arg7 = %8, %arg8 = %10, %arg9 = %12, %arg10 = %14, %arg11 = %darg0) -> (tensor<4x2x2xf64>, tensor<10x2xf64>, tensor<4x2xf64>, tensor<4x2xf64>, tensor<4xf64>, tensor<4xf64>) {
    %21 = arith.subi %c9, %arg5 : index
    %22 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %1) -> (tensor<4xf64>) {
      %47 = tensor.extract_slice %arg4[%21, 0] [1, 2] [1, 1] : tensor<10x2xf64> to tensor<2xf64>
      %48 = tensor.extract_slice %arg1[%arg12, 0] [1, 2] [1, 1] : tensor<4x2xf64> to tensor<2xf64>
      %49 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%47, %48 : tensor<2xf64>, tensor<2xf64>) outs(%47 : tensor<2xf64>) {
      ^bb0(%arg14: f64, %arg15: f64, %arg16: f64):
        %63 = arith.subf %arg14, %arg15 : f64
        linalg.yield %63 : f64
      } -> tensor<2xf64>
      %50 = tensor.extract_slice %3[%arg12, 0] [1, 2] [1, 1] : tensor<4x2xf64> to tensor<2xf64>
      %51 = tensor.extract_slice %arg3[%arg12, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<4x2x2xf64> to tensor<2x2xf64>
      %52 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%50, %49 : tensor<2xf64>, tensor<2xf64>) outs(%50 : tensor<2xf64>) {
      ^bb0(%arg14: f64, %arg15: f64, %arg16: f64):
        %63 = arith.mulf %arg14, %arg15 : f64
        linalg.yield %63 : f64
      } -> tensor<2xf64>
      %53 = linalg.matvec ins(%51, %49 : tensor<2x2xf64>, tensor<2xf64>) outs(%cst_4 : tensor<2xf64>) -> tensor<2xf64>
      %54 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%52, %53 : tensor<2xf64>, tensor<2xf64>) outs(%52 : tensor<2xf64>) {
      ^bb0(%arg14: f64, %arg15: f64, %arg16: f64):
        %63 = arith.addf %arg14, %arg15 : f64
        linalg.yield %63 : f64
      } -> tensor<2xf64>
      %55 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction"]} ins(%54 : tensor<2xf64>) outs(%cst_3 : tensor<f64>) {
      ^bb0(%arg14: f64, %arg15: f64):
        %63 = arith.mulf %arg14, %arg14 : f64
        %64 = arith.addf %63, %arg15 : f64
        linalg.yield %64 : f64
      } -> tensor<f64>
      %56 = tensor.extract %55[] : tensor<f64>
      %57 = arith.mulf %56, %cst_2 : f64
      %58 = tensor.extract %arg0[%arg12] : tensor<4xf64>
      %59 = tensor.extract %4[%arg12] : tensor<4xf64>
      %60 = arith.addf %58, %59 : f64
      %61 = arith.subf %60, %57 : f64
      %62 = tensor.insert %61 into %arg13[%arg12] : tensor<4xf64>
      scf.yield %62 : tensor<4xf64>
    }
    %23 = tensor.extract %22[%c0] : tensor<4xf64>
    %24 = tensor.insert %23 into %2[] : tensor<f64>
    %25 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction"]} ins(%22 : tensor<4xf64>) outs(%24 : tensor<f64>) {
    ^bb0(%arg12: f64, %arg13: f64):
      %47 = arith.cmpf ogt, %arg12, %arg13 : f64
      %48 = arith.select %47, %arg12, %arg13 : f64
      linalg.yield %48 : f64
    } -> tensor<f64>
    %26 = tensor.extract %25[] : tensor<f64>
    %27 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction"]} ins(%22 : tensor<4xf64>) outs(%cst_3 : tensor<f64>) {
    ^bb0(%arg12: f64, %arg13: f64):
      %47 = arith.subf %arg12, %26 : f64
      %48 = math.exp %47 : f64
      %49 = arith.addf %48, %arg13 : f64
      linalg.yield %49 : f64
    } -> tensor<f64>
    %28 = tensor.extract %27[] : tensor<f64>
    %29 = arith.divf %cst_1, %28 : f64
    %30 = linalg.init_tensor [] : tensor<f64>
    %31 = linalg.fill(%cst_0, %30) : f64, tensor<f64> -> tensor<f64> 
    %32 = tensor.insert %29 into %31[] : tensor<f64>
    %33 = linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["parallel"]} ins(%22, %32 : tensor<4xf64>, tensor<f64>) outs(%cst_3 : tensor<f64>) {
    ^bb0(%arg12: f64, %arg13: f64, %arg14: f64):
      %47 = arith.subf %arg12, %26 : f64
      %48 = math.exp %47 : f64
      %49 = arith.mulf %arg13, %48 : f64
      %50 = arith.negf %49 : f64
      %51 = arith.addf %50, %arg14 : f64
      linalg.yield %51 : f64
    } -> tensor<f64>
    %34 = tensor.extract %33[] : tensor<f64>
    %35 = arith.addf %34, %cst_1 : f64
    %36 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%22, %32 : tensor<4xf64>, tensor<f64>) outs(%cst_5 : tensor<4xf64>) {
    ^bb0(%arg12: f64, %arg13: f64, %arg14: f64):
      %47 = arith.subf %arg12, %26 : f64
      %48 = math.exp %47 : f64
      %49 = arith.mulf %arg13, %48 : f64
      linalg.yield %49 : f64
    } -> tensor<4xf64>
    %37 = linalg.init_tensor [] : tensor<f64>
    %38 = linalg.fill(%cst_0, %37) : f64, tensor<f64> -> tensor<f64> 
    %39 = tensor.insert %35 into %38[] : tensor<f64>
    %40 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%22, %39 : tensor<4xf64>, tensor<f64>) outs(%cst_5 : tensor<4xf64>) {
    ^bb0(%arg12: f64, %arg13: f64, %arg14: f64):
      %47 = arith.cmpf ogt, %arg12, %arg14 : f64
      %48 = arith.select %47, %arg13, %cst_0 : f64
      linalg.yield %48 : f64
    } -> tensor<4xf64>
    %41 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%36, %40 : tensor<4xf64>, tensor<4xf64>) outs(%36 : tensor<4xf64>) {
    ^bb0(%arg12: f64, %arg13: f64, %arg14: f64):
      %47 = arith.addf %arg12, %arg13 : f64
      linalg.yield %47 : f64
    } -> tensor<4xf64>
    %42 = tensor.extract %39[] : tensor<f64>
    %43 = tensor.extract %41[%c0] : tensor<4xf64>
    %44 = arith.addf %43, %42 : f64
    %45 = tensor.insert %44 into %41[%c0] : tensor<4xf64>
    %46:6 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg6, %arg14 = %arg7, %arg15 = %arg8, %arg16 = %arg9, %arg17 = %arg10, %arg18 = %arg11) -> (tensor<4x2x2xf64>, tensor<10x2xf64>, tensor<4x2xf64>, tensor<4x2xf64>, tensor<4xf64>, tensor<4xf64>) {
      %47 = arith.subi %c3, %arg12 : index
      %48 = tensor.extract_slice %arg4[%21, 0] [1, 2] [1, 1] : tensor<10x2xf64> to tensor<2xf64>
      %49 = tensor.extract_slice %arg1[%47, 0] [1, 2] [1, 1] : tensor<4x2xf64> to tensor<2xf64>
      %50 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%48, %49 : tensor<2xf64>, tensor<2xf64>) outs(%48 : tensor<2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.subf %arg19, %arg20 : f64
        linalg.yield %87 : f64
      } -> tensor<2xf64>
      %51 = tensor.extract_slice %3[%47, 0] [1, 2] [1, 1] : tensor<4x2xf64> to tensor<2xf64>
      %52 = tensor.extract_slice %arg3[%47, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<4x2x2xf64> to tensor<2x2xf64>
      %53 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%51, %50 : tensor<2xf64>, tensor<2xf64>) outs(%51 : tensor<2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.mulf %arg19, %arg20 : f64
        linalg.yield %87 : f64
      } -> tensor<2xf64>
      %54 = linalg.matvec ins(%52, %50 : tensor<2x2xf64>, tensor<2xf64>) outs(%cst_4 : tensor<2xf64>) -> tensor<2xf64>
      %55 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%53, %54 : tensor<2xf64>, tensor<2xf64>) outs(%53 : tensor<2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.addf %arg19, %arg20 : f64
        linalg.yield %87 : f64
      } -> tensor<2xf64>
      %56 = tensor.extract %45[%47] : tensor<4xf64>
      %57 = arith.negf %56 : f64
      %58 = tensor.extract %arg17[%47] : tensor<4xf64>
      %59 = arith.addf %58, %56 : f64
      %60 = tensor.insert %59 into %arg17[%47] : tensor<4xf64>
      %61 = tensor.extract %arg18[%47] : tensor<4xf64>
      %62 = arith.addf %61, %56 : f64
      %63 = tensor.insert %62 into %arg18[%47] : tensor<4xf64>
      %64 = arith.mulf %57, %cst_2 : f64
      %65 = linalg.init_tensor [] : tensor<f64>
      %66 = linalg.fill(%cst_0, %65) : f64, tensor<f64> -> tensor<f64> 
      %67 = tensor.insert %64 into %66[] : tensor<f64>
      %68 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%55, %67 : tensor<2xf64>, tensor<f64>) outs(%cst_4 : tensor<2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.mulf %arg20, %arg19 : f64
        %88 = arith.mulf %arg20, %arg19 : f64
        %89 = arith.addf %87, %88 : f64
        linalg.yield %89 : f64
      } -> tensor<2xf64>
      %69 = linalg.generic {doc = "Vector-vector outer product", indexing_maps = [#map1, #map4, #map0], iterator_types = ["parallel", "parallel"], library_call = "souter"} ins(%68, %50 : tensor<2xf64>, tensor<2xf64>) outs(%52 : tensor<2x2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.mulf %arg19, %arg20 : f64
        linalg.yield %87 : f64
      } -> tensor<2x2xf64>
      %70 = linalg.generic {doc = "Vector-Matrix multiplication", indexing_maps = [#map1, #map0, #map4], iterator_types = ["reduction", "parallel"], library_call = "svecmat"} ins(%68, %52 : tensor<2xf64>, tensor<2x2xf64>) outs(%cst_4 : tensor<2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.mulf %arg19, %arg20 : f64
        %88 = arith.addf %87, %arg21 : f64
        linalg.yield %88 : f64
      } -> tensor<2xf64>
      %71 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%68, %50 : tensor<2xf64>, tensor<2xf64>) outs(%68 : tensor<2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.mulf %arg19, %arg20 : f64
        linalg.yield %87 : f64
      } -> tensor<2xf64>
      %72 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%68, %51 : tensor<2xf64>, tensor<2xf64>) outs(%68 : tensor<2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.mulf %arg19, %arg20 : f64
        linalg.yield %87 : f64
      } -> tensor<2xf64>
      %73 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%70, %72 : tensor<2xf64>, tensor<2xf64>) outs(%70 : tensor<2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.addf %arg19, %arg20 : f64
        linalg.yield %87 : f64
      } -> tensor<2xf64>
      %74 = tensor.extract_slice %arg13[%47, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<4x2x2xf64> to tensor<2x2xf64>
      %75 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%74, %69 : tensor<2x2xf64>, tensor<2x2xf64>) outs(%74 : tensor<2x2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.addf %arg19, %arg20 : f64
        linalg.yield %87 : f64
      } -> tensor<2x2xf64>
      %76 = tensor.insert_slice %75 into %arg13[%47, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf64> into tensor<4x2x2xf64>
      %77 = tensor.extract_slice %arg15[%47, 0] [1, 2] [1, 1] : tensor<4x2xf64> to tensor<2xf64>
      %78 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%77, %71 : tensor<2xf64>, tensor<2xf64>) outs(%77 : tensor<2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.addf %arg19, %arg20 : f64
        linalg.yield %87 : f64
      } -> tensor<2xf64>
      %79 = tensor.insert_slice %78 into %arg15[%47, 0] [1, 2] [1, 1] : tensor<2xf64> into tensor<4x2xf64>
      %80 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%73 : tensor<2xf64>) outs(%73 : tensor<2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64):
        %87 = arith.negf %arg19 : f64
        linalg.yield %87 : f64
      } -> tensor<2xf64>
      %81 = tensor.extract_slice %arg16[%47, 0] [1, 2] [1, 1] : tensor<4x2xf64> to tensor<2xf64>
      %82 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%81, %80 : tensor<2xf64>, tensor<2xf64>) outs(%81 : tensor<2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.addf %arg19, %arg20 : f64
        linalg.yield %87 : f64
      } -> tensor<2xf64>
      %83 = tensor.insert_slice %82 into %arg16[%47, 0] [1, 2] [1, 1] : tensor<2xf64> into tensor<4x2xf64>
      %84 = tensor.extract_slice %arg14[%21, 0] [1, 2] [1, 1] : tensor<10x2xf64> to tensor<2xf64>
      %85 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%84, %73 : tensor<2xf64>, tensor<2xf64>) outs(%84 : tensor<2xf64>) {
      ^bb0(%arg19: f64, %arg20: f64, %arg21: f64):
        %87 = arith.addf %arg19, %arg20 : f64
        linalg.yield %87 : f64
      } -> tensor<2xf64>
      %86 = tensor.insert_slice %85 into %arg14[%21, 0] [1, 2] [1, 1] : tensor<2xf64> into tensor<10x2xf64>
      scf.yield %76, %86, %79, %83, %60, %63 : tensor<4x2x2xf64>, tensor<10x2xf64>, tensor<4x2xf64>, tensor<4x2xf64>, tensor<4xf64>, tensor<4xf64>
    }
    scf.yield %46#0, %46#1, %46#2, %46#3, %46#4, %46#5 : tensor<4x2x2xf64>, tensor<10x2xf64>, tensor<4x2xf64>, tensor<4x2xf64>, tensor<4xf64>, tensor<4xf64>
  }
  %18 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %17#4 : tensor<4x2xf64>, tensor<4xf64>) outs(%cst : tensor<4x2xf64>) {
  ^bb0(%arg5: f64, %arg6: f64, %arg7: f64):
    linalg.yield %arg6 : f64
  } -> tensor<4x2xf64>
  %19 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %17#2 : tensor<4x2xf64>, tensor<4x2xf64>) outs(%cst : tensor<4x2xf64>) {
  ^bb0(%arg5: f64, %arg6: f64, %arg7: f64):
    %21 = math.exp %arg5 : f64
    %22 = arith.mulf %arg6, %21 : f64
    linalg.yield %22 : f64
  } -> tensor<4x2xf64>
  // %20 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : tensor<4x2xf64>, tensor<4x2xf64>) outs(%darg2 : tensor<4x2xf64>) {
  // ^bb0(%arg5: f64, %arg6: f64, %arg7: f64):
  //   %21 = arith.addf %arg5, %arg6 : f64
  //   linalg.yield %21 : f64
  // } -> tensor<4x2xf64>
  return %17#5 : tensor<4xf64>
  // return %17#5, %17#3, %20, %17#0 : tensor<4xf64>, tensor<4x2xf64>, tensor<4x2xf64>, tensor<4x2x2xf64>
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

  // %res = call @main_term(%alphas, %means, %Qs, %Ls, %x) :  (tensor<4xf64>, tensor<4x2xf64>, tensor<4x2xf64>, tensor<4x2x2xf64>, tensor<10x2xf64>) -> f64
  // %U = linalg.init_tensor [] : tensor<f64>
  // %U0 = tensor.insert %res into %U[] : tensor<f64>
  // %U1 = tensor.cast %U0 : tensor<f64> to tensor<*xf64>
  // call @print_memref_f64(%U1) : (tensor<*xf64>) -> ()

  // %f = constant @main_term : (tensor<4xf64>, tensor<4x2xf64>, tensor<4x2xf64>, tensor<4x2x2xf64>, tensor<10x2xf64>) -> f64
  // %df = standalone.grad %f {of = [0, 1, 2, 3]} : (
  //   tensor<4xf64>,
  //   tensor<4x2xf64>,
  //   tensor<4x2xf64>,
  //   tensor<4x2x2xf64>,
  //   tensor<10x2xf64>
  // ) -> f64, (
  //   tensor<4xf64>,
  //   tensor<4x2xf64>,
  //   tensor<4x2xf64>,
  //   tensor<4x2x2xf64>,
  //   tensor<10x2xf64>    
  // ) -> (
  //   tensor<4xf64>,
  //   tensor<4x2xf64>,
  //   tensor<4x2xf64>,
  //   tensor<4x2x2xf64>
  // )
  // %res:4 = call_indirect %df(%alphas, %means, %Qs, %Ls, %x) : (
  //   tensor<4xf64>,
  //   tensor<4x2xf64>,
  //   tensor<4x2xf64>,
  //   tensor<4x2x2xf64>,
  //   tensor<10x2xf64>    
  // ) -> (
  //   tensor<4xf64>,
  //   tensor<4x2xf64>,
  //   tensor<4x2xf64>,
  //   tensor<4x2x2xf64>
  // )

  // %U0 = tensor.cast %res#0 : tensor<4xf64> to tensor<*xf64>
  // call @print_memref_f64(%U0) : (tensor<*xf64>) -> ()
  // %U1 = tensor.cast %res#1 : tensor<4x2xf64> to tensor<*xf64>
  // call @print_memref_f64(%U1) : (tensor<*xf64>) -> ()
  // %U2 = tensor.cast %res#2 : tensor<4x2xf64> to tensor<*xf64>
  // call @print_memref_f64(%U2) : (tensor<*xf64>) -> ()
  // %U3 = tensor.cast %res#3 : tensor<4x2x2xf64> to tensor<*xf64>
  // call @print_memref_f64(%U3) : (tensor<*xf64>) -> ()

  %dres = linalg.init_tensor [4] : tensor<4xf64>
  %res = call @__grad_main_term(%alphas, %means, %Qs, %Ls, %x, %dres) : (
    tensor<4xf64>,
    tensor<4x2xf64>,
    tensor<4x2xf64>,
    tensor<4x2x2xf64>,
    tensor<10x2xf64>,
    tensor<4xf64>
  ) -> tensor<4xf64>
  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
