#map8 = affine_map<(d0, d1) -> (d0, d1)>
#map9 = affine_map<(d0, d1) -> (d0)>

func @colhot_broadcast_mul(%6: tensor<3xf64>, %15: tensor<3x3xf64, "colhot">) -> tensor<3x3xf64> {
  %cst_2 = arith.constant dense<0.0> : tensor<3x3xf64>
  %17 = linalg.generic
    { indexing_maps = [#map9, #map8, #map8], iterator_types = ["parallel", "parallel"] }
    ins(%6, %15 : tensor<3xf64>, tensor<3x3xf64, "colhot">)
    outs(%cst_2 : tensor<3x3xf64>) {
  ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
    %38 = arith.mulf %arg10, %arg9 : f64
    %39 = arith.addf %38, %arg11 : f64
    linalg.yield %39 : f64
  } -> tensor<3x3xf64>
  return %17 : tensor<3x3xf64>
}
