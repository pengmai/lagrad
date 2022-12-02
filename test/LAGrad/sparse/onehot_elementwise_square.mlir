#map = affine_map<(d0, d1) -> (d0, d1)>
func @onehot_square(%arg: tensor<10x3xf64, "onehot">) -> tensor<10x3xf64> {
  %space = linalg.init_tensor [10, 3] : tensor<10x3xf64>
  %zero = arith.constant 0.0 : f64
  %dest = linalg.fill(%zero, %space) : f64, tensor<10x3xf64> -> tensor<10x3xf64>
  %res = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg : tensor<10x3xf64, "onehot">) outs(%dest : tensor<10x3xf64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = arith.mulf %arg0, %arg0 : f64
    // %0 = arith.addf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  } -> tensor<10x3xf64>
  return %res : tensor<10x3xf64>
}
