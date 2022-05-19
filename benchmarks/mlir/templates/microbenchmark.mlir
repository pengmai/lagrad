#map = affine_map<(d0) -> (d0)>
func @dot(%arg0: tensor<{{n}}xf64>, %arg1: tensor<{{n}}xf64>) -> tensor<f64> {
  %zero = arith.constant dense<0.0> : tensor<f64>
  %res = linalg.dot ins(%arg0, %arg1 : tensor<{{n}}xf64>, tensor<{{n}}xf64>) outs(%zero : tensor<f64>) -> tensor<f64>
  return %res : tensor<f64>
}

func @lagrad_dot(%arg0: tensor<{{n}}xf64>, %arg1: tensor<{{n}}xf64>) -> tensor<{{n}}xf64> {
  %f = constant @dot : (tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<f64>
  %df = standalone.grad %f {of = [0]} : (tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<f64>, (tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  %res = call_indirect %df(%arg0, %arg1) : (tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  return %res : tensor<{{n}}xf64>
}

// #map0 = affine_map<(d0) -> ()>
// #map1 = affine_map<(d0) -> (d0)>
// memref.global "private" constant @__constant_16384xf64 : memref<16384xf64> = dense<0.000000e+00>
// memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
// func @__grad_mvectorscalar(%arg0: memref<16384xf64>, %arg1: memref<f64>, %arg2: memref<16384xf64>) -> memref<f64> {
//   %0 = memref.get_global @__constant_xf64 : memref<f64>
//   %1 = memref.alloc() : memref<f64>
//   linalg.copy(%0, %1) : memref<f64>, memref<f64>
//   linalg.generic {indexing_maps = [#map1, #map1, #map0], iterator_types = ["parallel"]} ins(%arg0, %arg2 : memref<16384xf64>, memref<16384xf64>) outs(%1 : memref<f64>) {
//   ^bb0(%arg3: f64, %arg4: f64, %arg5: f64):  // no predecessors
//     %2 = arith.mulf %arg4, %arg3 : f64
//     %3 = arith.addf %2, %arg5 : f64
//     linalg.yield %3 : f64
//   }
//   return %1 : memref<f64>
// }
// func @lagrad_vecscal(%arg0: memref<16384xf64>, %arg1: f64, %arg2: memref<16384xf64>) -> f64 {
//   %0 = memref.alloca() : memref<f64>
//   memref.store %arg1, %0[] : memref<f64>
//   %1 = call @__grad_mvectorscalar(%arg0, %0, %arg2) : (memref<16384xf64>, memref<f64>, memref<16384xf64>) -> memref<f64>
//   %2 = memref.load %1[] : memref<f64>
//   return %2 : f64
// }
func @mvectorscalar(%arg0: tensor<{{n}}xf64>, %arg1: tensor<f64>) -> tensor<{{n}}xf64> {
  %space = arith.constant dense<0.0> : tensor<{{n}}xf64>
  %res = linalg.generic
    {
      indexing_maps = [#map, affine_map<(d0) -> ()>, #map],
      iterator_types = ["parallel"]
    }
    ins(%arg0, %arg1 : tensor<{{n}}xf64>, tensor<f64>)
    outs(%space : tensor<{{n}}xf64>) {
  ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
    %0 = arith.mulf %arg2, %arg3 : f64
    linalg.yield %0 : f64
  } -> tensor<{{n}}xf64>
  return %res : tensor<{{n}}xf64>
}

func @lagrad_vecscal(%arg0: tensor<{{n}}xf64>, %arg1: f64, %g: tensor<{{n}}xf64>) -> f64 {
  %arg1_space = linalg.init_tensor [] : tensor<f64>
  %arg1t = tensor.insert %arg1 into %arg1_space[] : tensor<f64>
  %f = constant @mvectorscalar : (tensor<{{n}}xf64>, tensor<f64>) -> tensor<{{n}}xf64>
  %df = standalone.grad %f {of = [1], grad_signal = true} :
    (tensor<{{n}}xf64>, tensor<f64>) -> tensor<{{n}}xf64>,
    (tensor<{{n}}xf64>, tensor<f64>, tensor<{{n}}xf64>) -> tensor<f64>
  %res = call_indirect %df(%arg0, %arg1t, %g) : (tensor<{{n}}xf64>, tensor<f64>, tensor<{{n}}xf64>) -> tensor<f64>
  %res_val = tensor.extract %res[] : tensor<f64>
  return %res_val : f64
}
