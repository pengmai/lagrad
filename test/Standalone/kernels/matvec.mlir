func @matvec(%M : tensor<512x1024xf32>, %x : tensor<1024xf32>) -> tensor<512xf32> {
  %dummy = constant dense<0.0> : tensor<512xf32>
  %res = linalg.matvec ins(%M, %x : tensor<512x1024xf32>, tensor<1024xf32>) outs(%dummy : tensor<512xf32>) -> tensor<512xf32>
  return %res : tensor<512xf32>
}

// func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

// #map0 = affine_map<(d0, d1) -> (d0)>
// #map1 = affine_map<(d0, d1) -> (d0, d1)>
// #map2 = affine_map<(d0, d1) -> (d1)>
// func @sumreduce(%g : tensor<512xf32>, %M : tensor<512x1024xf32>) -> tensor<*xf32> {
//   %out = constant dense<0.0> : tensor<1024xf32>
//   %reduced = linalg.generic
//     {indexing_maps = [#map0, #map1, #map2], iterator_types = ["reduction", "parallel"]}
//     ins(%g, %M : tensor<512xf32>, tensor<512x1024xf32>)
//     outs(%out : tensor<1024xf32>) {
//   ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
//     %0 = mulf %arg2, %arg3 : f32
//     %1 = addf %0, %arg4 : f32
//     linalg.yield %1 : f32
//   } -> tensor<1024xf32>
//   %casted = tensor.cast %reduced : tensor<1024xf32> to tensor<*xf32>
//   return %casted : tensor<*xf32>
// }

func @grad_matvec(%M : tensor<512x1024xf32>, %x : tensor<1024xf32>) -> (tensor<512x1024xf32>, tensor<1024xf32>) {
  %f = constant @matvec : (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512xf32>
  %df = standalone.grad %f {of=[0, 1]}: (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512xf32>, (tensor<512x1024xf32>, tensor<1024xf32>) -> (tensor<512x1024xf32>, tensor<1024xf32>)
  %res:2 = call_indirect %df(%M, %x) : (tensor<512x1024xf32>, tensor<1024xf32>) -> (tensor<512x1024xf32>, tensor<1024xf32>)
  return %res#0, %res#1 : tensor<512x1024xf32>, tensor<1024xf32>
  // %casted = tensor.cast %res#0 : tensor<512x1024xf32> to tensor<*xf32>
  // %casted2 = tensor.cast %res#1 : tensor<1024xf32> to tensor<*xf32>
  // return %casted, %casted2 : tensor<*xf32>, tensor<*xf32>
}

// func @main() {
//   %M = constant dense<1.2> : tensor<512x1024xf32>
//   %x = constant dense<1.1> : tensor<1024xf32>
// }

// --- Reference ---
