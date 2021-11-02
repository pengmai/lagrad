// An elementwise gradient of a matrix-vector multiplication.
func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

func @matvec(%arg0 : tensor<3x4xf32>, %arg1 : tensor<4xf32>) -> tensor<3xf32> {
  %res = arith.constant dense<0.0> : tensor<3xf32>
  %val = linalg.matvec ins(%arg0, %arg1 : tensor<3x4xf32>, tensor<4xf32>) outs(%res : tensor<3xf32>) -> tensor<3xf32>
  %c = arith.constant dense<[1.1, -1.2, 1.0]> : tensor<3xf32>
  %final = arith.mulf %val, %c : tensor<3xf32>
  return %final : tensor<3xf32>
}

// #map0 = affine_map<(d0, d1) -> (d0)>
// #map1 = affine_map<(d0, d1) -> (d1)>
// #map2 = affine_map<(d0, d1) -> (d0, d1)>
// func @mygrad(%arg0 : tensor<3x4xf32>, %arg1 : tensor<4xf32>) -> tensor<3x4xf32> {
//   %grad_signal = constant dense<[1.1, -1.2, 1.0]> : tensor<3xf32>
//   %output = constant dense<0.0> : tensor<3x4xf32>
//   %res = linalg.generic
//     {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel"]}
//     ins(%grad_signal, %arg1 : tensor<3xf32>, tensor<4xf32>)
//     outs(%output : tensor<3x4xf32>) {
//   ^bb0(%arg2 : f32, %arg3 : f32, %arg4 : f32):
//     %0 = mulf %arg2, %arg3 : f32
//     linalg.yield %0 : f32
//   } -> tensor<3x4xf32>
//   return %res : tensor<3x4xf32>
// }

func @main() {
  %M = arith.constant dense<[
    [0.8,  0.9,  1.2, -4.3],
    [2.3, -1.1, -7.5, -3.2],
    [1.2,  1.1,  1.0,  0.0]
  ]> : tensor<3x4xf32>
  %x = arith.constant dense<[-1.2, -1.3, 1.5, 2.2]> : tensor<4xf32>

  %f = constant @matvec : (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3xf32>
  %df = standalone.grad %f {of=[0]} : (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3xf32>, (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  %val = call_indirect %df(%M, %x) : (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  // %val = call @mygrad(%M, %x) : (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3x4xf32>

  %casted = tensor.cast %val : tensor<3x4xf32> to tensor<*xf32>
  call @print_memref_f32(%casted) : (tensor<*xf32>) -> ()
  return
}
