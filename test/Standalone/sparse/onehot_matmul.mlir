#map5 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map6 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map10 = affine_map<(d0, d1, d2) -> (d2, d0)>

// What is matmul usually?
// mk * kn -> mn
// this is:
// km * kn -> nm
// The result should be column-hot.
func @onehot_matmul(%4: tensor<544x3xf64>, %12: tensor<544x3xf64, "onehot">) -> tensor<3x3xf64> {
  %cst_2 = arith.constant dense<0.0> : tensor<3x3xf64>
  %15 = linalg.generic {indexing_maps = [#map6, #map5, #map10], iterator_types = ["parallel", "reduction", "parallel"]} ins(%4, %12 : tensor<544x3xf64>, tensor<544x3xf64, "onehot">) outs(%cst_2 : tensor<3x3xf64>) {
  ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
    %38 = arith.mulf %arg10, %arg9 : f64
    %39 = arith.addf %38, %arg11 : f64
    linalg.yield %39 : f64
  } -> tensor<3x3xf64>
  return %15 : tensor<3x3xf64>
}

// both args are transposed
// result is rowhot
// [c0, ln3, iv]
func @onehot_matmul_both_transposed(%8: tensor<3x3xf64>, %12: tensor<544x3xf64, "onehot">) -> tensor<544x3xf64> {
  %cst_3 = arith.constant dense<0.0> : tensor<544x3xf64>
  %16 = linalg.generic {indexing_maps = [#map10, #map5, #map6], iterator_types = ["reduction", "parallel", "parallel"]} ins(%8, %12 : tensor<3x3xf64>, tensor<544x3xf64, "onehot">) outs(%cst_3 : tensor<544x3xf64>) {
  ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
    %38 = arith.mulf %arg10, %arg9 : f64
    %39 = arith.addf %38, %arg11 : f64
    linalg.yield %39 : f64
  } -> tensor<544x3xf64>
  return %16 : tensor<544x3xf64>
}
