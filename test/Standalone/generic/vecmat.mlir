#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>

func.func @vecmat(%arg0: tensor<3xf32>, %arg1: tensor<3x4xf32>) -> tensor<4xf32> {
  %cst = arith.constant dense<0.0> : tensor<4xf32>
  %0 = linalg.generic
    {
      indexing_maps = [#map0, #map1, #map2],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%arg0, %arg1 : tensor<3xf32>, tensor<3x4xf32>)
    outs(%cst : tensor<4xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
    %1 = arith.mulf %arg2, %arg3 : f32
    %2 = arith.addf %arg4, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func private @printMemrefF32(tensor<*xf32>) attributes { llvm.emit_c_interface }

func.func @main() {
  %arg0 = arith.constant dense<[1., 2., 3.]> : tensor<3xf32>
  %arg1 = arith.constant dense<[[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]]> : tensor<3x4xf32>
  %f = constant @vecmat : (tensor<3xf32>, tensor<3x4xf32>) -> tensor<4xf32>
  %df = standalone.grad %f {of = [0]}: (tensor<3xf32>, tensor<3x4xf32>) -> tensor<4xf32>, (tensor<3xf32>, tensor<3x4xf32>) -> tensor<3xf32>
  %res = call_indirect %df(%arg0, %arg1) : (tensor<3xf32>, tensor<3x4xf32>) -> tensor<3xf32>
  %U = tensor.cast %res : tensor<3xf32> to tensor<*xf32>
  call @printMemrefF32(%U) : (tensor<*xf32>) -> ()
  return
}
