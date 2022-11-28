#map8 = affine_map<(d0, d1) -> (d0, d1)>
#map11 = affine_map<(d0, d1) -> (d1)>

func @onehot_sumreduce(%12: tensor<544x3xf64, "onehot">) -> tensor<3xf64> {
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<3xf64>
  %14 = linalg.generic {indexing_maps = [#map8, #map11], iterator_types = ["reduction", "parallel"]} ins(%12 : tensor<544x3xf64, "onehot">) outs(%cst_1 : tensor<3xf64>) {
  ^bb0(%arg9: f64, %arg10: f64):  // no predecessors
    %38 = arith.addf %arg9, %arg10 : f64
    linalg.yield %38 : f64
  } -> tensor<3xf64>
  return %14 : tensor<3xf64>
}
