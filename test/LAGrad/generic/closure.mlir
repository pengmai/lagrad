#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<() -> ()>
func.func @closure(%freevar: f32, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.constant dense<0.0> : tensor<4xf32>
  %1 = linalg.generic
    {
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel"]
    }
    ins(%arg1, %arg2 : tensor<4xf32>, tensor<4xf32>)
    outs(%0 : tensor<4xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.mulf %arg3, %freevar : f32
    %3 = arith.mulf %2, %arg4 : f32
    linalg.yield %3: f32
  } -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

func.func private @printMemrefF32(tensor<*xf32>) attributes { llvm.emit_c_interface }

func.func @main() {
  %cst = arith.constant 4.4 : f32
  %a = arith.constant dense<[1., 2., 3., 4.]> : tensor<4xf32>
  %b = arith.constant dense<[2., 3., 4., 5.]> : tensor<4xf32>

  %res = lagrad.grad @closure(%cst, %a, %b) : (f32, tensor<4xf32>, tensor<4xf32>) -> f32

  %space = arith.constant dense<0.0> : tensor<f32>
  %inserted = tensor.insert %res into %space[] : tensor<f32>
  %U = tensor.cast %inserted : tensor<f32> to tensor<*xf32>
  call @printMemrefF32(%U) : (tensor<*xf32>) -> ()
  return
}
