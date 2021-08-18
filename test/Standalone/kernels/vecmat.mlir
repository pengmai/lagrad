func @vecmat(%x : tensor<3xf32>, %M : tensor<3x4xf32>) -> tensor<4xf32> {
  %shape = constant dense<0.0> : tensor<4xf32>
  %res = linalg.vecmat ins(%x, %M : tensor<3xf32>, tensor<3x4xf32>) outs(%shape : tensor<4xf32>) -> tensor<4xf32>
  %cst = constant dense<[2.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
  %final = mulf %res, %cst : tensor<4xf32>
  return %final : tensor<4xf32>
}

// #map0 = affine_map<(d0, d1) -> (d1)>
// #map1 = affine_map<(d0, d1) -> (d0, d1)>
// #map2 = affine_map<(d0, d1) -> (d0)>
// func @__grad_vecmat(%x : tensor<3xf32>, %M : tensor<3x4xf32>) -> tensor<3xf32> {
//   %shape = constant dense<0.0> : tensor<3xf32>
//   %g = constant dense<[2.0, 1.0, 1.0, 1.0]> : tensor<4xf32>

//   %res = linalg.generic
//     {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "reduction"]}
// 	ins(%g, %M : tensor<4xf32>, tensor<3x4xf32>)
// 	outs(%shape : tensor<3xf32>) {
//   ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
//     %0 = mulf %arg0, %arg1 : f32
// 	%1 = addf %0, %arg2 : f32
// 	linalg.yield %1 : f32
//   } -> tensor<3xf32>
//   return %res : tensor<3xf32>
// }

func @grad_vecmat(%x : tensor<3xf32>, %M : tensor<3x4xf32>) -> tensor<3x4xf32> {
  %f = constant @vecmat : (tensor<3xf32>, tensor<3x4xf32>) -> tensor<4xf32>
  %df = standalone.grad %f {of = [1]} : (tensor<3xf32>, tensor<3x4xf32>) -> tensor<4xf32>, (tensor<3xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  %res = call_indirect %df(%x, %M) : (tensor<3xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
//   %res = call @__grad_vecmat(%x, %M) : (tensor<3xf32>, tensor<3x4xf32>) -> tensor<3xf32>
  return %res : tensor<3x4xf32>
}
