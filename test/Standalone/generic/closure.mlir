#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<() -> ()>
func @closure(%arg0: f32, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = constant dense<0.0> : tensor<4xf32>
  %1 = linalg.generic
    {
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel"]
    }
    ins(%arg1, %arg2 : tensor<4xf32>, tensor<4xf32>)
    outs(%0 : tensor<4xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = mulf %arg3, %arg0 : f32
    %3 = mulf %2, %arg4 : f32
    linalg.yield %3: f32
  } -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

func @main() {
  %cst = constant 4.4 : f32
  %a = constant dense<[1., 2., 3., 4.]> : tensor<4xf32>
  %b = constant dense<[2., 3., 4., 5.]> : tensor<4xf32>

  %f = constant @closure : (f32, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %df = standalone.grad %f {of = [0]}: (f32, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>, (f32, tensor<4xf32>, tensor<4xf32>) -> f32
  %res = call_indirect %df(%cst, %a, %b) : (f32, tensor<4xf32>, tensor<4xf32>) -> f32

  %space = constant dense<0.0> : tensor<f32>
  %inserted = tensor.insert %res into %space[] : tensor<f32>
  %U = tensor.cast %inserted : tensor<f32> to tensor<*xf32>
  call @print_memref_f32(%U) : (tensor<*xf32>) -> ()
  return
}
