func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

func @dot(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
  %cst = constant dense<0.0> : tensor<f32>
  %0 = linalg.generic
    {
      indexing_maps = [#map0, #map0, #map1],
      iterator_types = ["reduction"]
    }
    ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
    outs(%cst : tensor<f32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %1 = mulf %arg2, %arg3 : f32
    %2 = addf %1, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}

func @main() {
  %cst = constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %cst_0 = constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>

  %f = constant @dot : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
  %df = standalone.grad %f : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>, (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  %0 = call_indirect %df(%cst, %cst_0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %U = tensor.cast %0 : tensor<4xf32> to tensor<*xf32>
  call @print_memref_f32(%U) : (tensor<*xf32>) -> ()

  // %1 = call @handwritten_sub(%cst, %cst_0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // %U_0 = tensor.cast %1 : tensor<4xf32> to tensor<*xf32>
  // call @print_memref_f32(%U_0) : (tensor<*xf32>) -> ()
  return
}
