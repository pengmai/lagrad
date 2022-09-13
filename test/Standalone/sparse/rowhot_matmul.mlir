#map5 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map6 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map10 = affine_map<(d0, d1, d2) -> (d2, d0)>

// first arg is transposed
// Result is dense
// func @rowhot_matmul(%arg4: tensor<544x4xf64>, %43: tensor<544x4xf64, "rowhot">) -> tensor<4x4xf64> {
//   %space = linalg.init_tensor [4, 4] : tensor<4x4xf64>
//   %zero = arith.constant 0.0 : f64
//   %44 = linalg.fill(%zero, %space) : f64, tensor<4x4xf64> -> tensor<4x4xf64>
//   %45 = linalg.generic {indexing_maps = [#map6, #map5, #map10], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg4, %43 : tensor<544x4xf64>, tensor<544x4xf64, "rowhot">) outs(%44 : tensor<4x4xf64>) {
//   ^bb0(%arg11: f64, %arg12: f64, %arg13: f64):  // no predecessors
//     %47 = arith.mulf %arg12, %arg11 : f64
//     %48 = arith.addf %47, %arg13 : f64
//     linalg.yield %48 : f64
//   } -> tensor<4x4xf64>
//   return %45 : tensor<4x4xf64>
// }
