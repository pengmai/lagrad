func.func private @printMemrefF32(tensor<*xf32>) attributes { llvm.emit_c_interface }

#map = affine_map<(d0) -> (d0)>

func.func @sub(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.constant dense<0.0> : tensor<4xf32>
  %1 = linalg.generic
    {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel"]
    }
    ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
    outs(%0 : tensor<4xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %2 = arith.subf %arg2, %arg3 : f32
    linalg.yield %2 : f32
  } -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

func.func @main() {
  %cst = arith.constant dense<-1.3> : tensor<4xf32>
  %cst_0 = arith.constant dense<2.2> : tensor<4xf32>

  %f = constant @sub : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %df = standalone.grad %f {of = [1]} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>, (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  %res = call_indirect %df(%cst, %cst_0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %U = tensor.cast %res : tensor<4xf32> to tensor<*xf32>
  call @printMemrefF32(%U) : (tensor<*xf32>) -> ()
  return
}
