func @matmul(%A : tensor<3x4xf32>, %B : tensor<4x5xf32>) -> tensor<3x5xf32> {
  %out = arith.constant dense<0.0> : tensor<3x5xf32>
  %res = linalg.matmul ins(%A, %B : tensor<3x4xf32>, tensor<4x5xf32>) outs(%out : tensor<3x5xf32>) -> tensor<3x5xf32>
  return %res : tensor<3x5xf32>
}

// Gradient of first argument
// #map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
// #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// #map2 = affine_map<(d0, d1, d2) -> (d0, d2)>

func @grad_matmul(%A : tensor<3x4xf32>, %B : tensor<4x5xf32>) -> tensor<3x4xf32> {
  %f = constant @matmul : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
  %df = standalone.grad %f : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>, (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x4xf32>
  %res = call_indirect %df(%A, %B) : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x4xf32>

  // %g = constant dense<1.0> : tensor<3x5xf32>
  // %out = constant dense<0.0> : tensor<3x4xf32>
  // %res = linalg.generic
  //   {indexing_maps=[#map0, #map1, #map2], iterator_types=["parallel", "reduction", "parallel"]}
  //   ins(%g, %B : tensor<3x5xf32>, tensor<4x5xf32>)
  //   outs(%out : tensor<3x4xf32>) {
  // ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
  //   %0 = mulf %arg0, %arg1 : f32
  //   %1 = addf %arg2, %0 : f32
  //   linalg.yield %1 : f32
  // } -> tensor<3x4xf32>
  return %res : tensor<3x4xf32>
}
