func @matvec(%M : tensor<3x4xf32>, %x : tensor<4xf32>) -> tensor<3xf32> {
  %dummy = constant dense<0.0> : tensor<3xf32>
  %res = linalg.matvec ins(%M, %x : tensor<3x4xf32>, tensor<4xf32>) outs(%dummy : tensor<3xf32>) -> tensor<3xf32>
  return %res : tensor<3xf32>
}

// func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
func @sumreduce(%g : tensor<3xf32>, %M : tensor<3x4xf32>) -> tensor<*xf32> {
  %out = constant dense<0.0> : tensor<4xf32>
  %reduced = linalg.generic
    {indexing_maps = [#map0, #map1, #map2], iterator_types = ["reduction", "parallel"]}
    ins(%g, %M : tensor<3xf32>, tensor<3x4xf32>)
    outs(%out : tensor<4xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %0 = mulf %arg2, %arg3 : f32
    %1 = addf %0, %arg4 : f32
    linalg.yield %1 : f32
  } -> tensor<4xf32>
  %casted = tensor.cast %reduced : tensor<4xf32> to tensor<*xf32>
  return %casted : tensor<*xf32>
}

func @grad_matvec(%M : tensor<3x4xf32>, %x : tensor<4xf32>) -> tensor<*xf32> {
  %f = constant @matvec : (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3xf32>
  %df = standalone.grad %f : (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3xf32>, (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  %res = call_indirect %df(%M, %x) : (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  %casted = tensor.cast %res : tensor<3x4xf32> to tensor<*xf32>
  return %casted : tensor<*xf32>
}

// func @main() {
//   %M = constant dense<1.2> : tensor<3x4xf32>
//   %x = constant dense<1.1> : tensor<4xf32>
// }

// --- Reference ---
