func @matmul(%A : tensor<3x4xf32>, %B : tensor<4x5xf32>) -> tensor<3x5xf32> {
  %out = constant dense<0.0> : tensor<3x5xf32>
  %res = linalg.matmul ins(%A, %B : tensor<3x4xf32>, tensor<4x5xf32>) outs(%out : tensor<3x5xf32>) -> tensor<3x5xf32>
  return %res : tensor<3x5xf32>
}

func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

func @main() {
  %A = constant dense<[
    [ 6.9, -5.0,  3.3, -0.3],
    [ 2.9,  3.9, -1.1, -6.4],
    [-1.4,  0.2, -2.8,  8.5]
  ]> : tensor<3x4xf32>
  %B = constant dense<[
    [0.05205203, 0.79496252, 0.33748601, 0.0866164 , 0.42671222],
    [0.78915215, 0.7872005 , 0.56147273, 0.99961059, 0.63826415],
    [0.07791169, 0.7308148 , 0.97226241, 0.80065608, 0.46255539],
    [0.66985071, 0.08788304, 0.4886088 , 0.80950611, 0.00851334]
  ]> : tensor<4x5xf32>
  %f = constant @matmul : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>

  %df = standalone.grad %f : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>, (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x4xf32>
  %first = call_indirect %df(%A, %B) : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x4xf32>
  %casted = tensor.cast %first : tensor<3x4xf32> to tensor<*xf32>

  // %df = standalone.grad %f {of = [1]} : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>, (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  // %snd = call_indirect %df(%A, %B) : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  // %casted = tensor.cast %snd : tensor<4x5xf32> to tensor<*xf32>

  call @print_memref_f32(%casted) : (tensor<*xf32>) -> ()
  return
}

// Gradient of second argument
// #map0 = affine_map<(d0, d1, d2) -> (d1, d0)>
// #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
// #map2 = affine_map<(d0, d1, d2) -> (d0, d2)>

// func @grad_matmul(%A : tensor<3x4xf32>, %B : tensor<4x5xf32>) -> tensor<4x5xf32> {
//   // %f = constant @matmul : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
//   // %df = standalone.grad %f : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>, (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x4xf32>
//   // %res = call_indirect %df(%A, %B) : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x4xf32>

//   %g = constant dense<1.0> : tensor<3x5xf32>
//   %out = constant dense<0.0> : tensor<4x5xf32>

//   %res = linalg.generic
//     {indexing_maps=[#map0, #map1, #map2], iterator_types=["parallel", "reduction", "parallel"]}
//     ins(%A, %g : tensor<3x4xf32>, tensor<3x5xf32>)
//     outs(%out : tensor<4x5xf32>) {
//   ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
//     %0 = mulf %arg0, %arg1 : f32
//     %1 = addf %arg2, %0 : f32
//     linalg.yield %1 : f32
//   } -> tensor<4x5xf32>
//   return %res : tensor<4x5xf32>
// }