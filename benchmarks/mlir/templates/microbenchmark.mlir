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

// func @arrmax(%arg0: tensor<{{n}}xf64>) -> f64 {
//   %c0 = arith.constant 0 : index
//   %init_val = tensor.extract %arg0[%c0] : tensor<{{n}}xf64>
//   %init_space = linalg.init_tensor [] : tensor<f64>
//   %init = tensor.insert %init_val into %init_space[] : tensor<f64>
//   %res = linalg.generic
//     {
//       indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
//       iterator_types = ["reduction"]
//     }
//     ins(%arg0 : tensor<{{n}}xf64>)
//     outs(%init : tensor<f64>) {
//   ^bb0(%arg1: f64, %arg2: f64):
//     %p = arith.cmpf ogt, %arg1, %arg2 : f64
//     %0 = scf.if %p -> (f64) {
//       scf.yield %arg1 : f64
//     } else {
//       scf.yield %arg2 : f64
//     }
//     linalg.yield %0 : f64
//   } -> tensor<f64>
//   %res_val = tensor.extract %res[] : tensor<f64>
//   return %res_val : f64
// }

func @arrmax(%arg0: tensor<{{n}}xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cn = arith.constant {{n}} : index
  // %init = tensor.extract %arg0[%c0] : tensor<{{n}}xf64>
  %init = arith.constant -1.0 : f64
  %res = scf.for %iv = %c0 to %cn step %c1 iter_args(%iter_max = %init) -> f64 {
    %0 = tensor.extract %arg0[%iv] : tensor<{{n}}xf64>
    %p = arith.cmpf ogt, %0, %iter_max : f64
    %iter_next = scf.if %p -> (f64) {
      scf.yield %0 : f64
    } else {
      scf.yield %iter_max : f64
    }
    scf.yield %iter_next : f64
  }
  return %res : f64
}

func @lagrad_arrmax(%arg0: tensor<{{n}}xf64>) -> tensor<{{n}}xf64> {
  %f = constant @arrmax : (tensor<{{n}}xf64>) -> f64
  %df = standalone.grad %f {of = [0]} : (tensor<{{n}}xf64>) -> f64, (tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  %res = call_indirect %df(%arg0) : (tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  return %res : tensor<{{n}}xf64>
}

func @vecadd(%arg0: tensor<{{n}}xf64>, %arg1: tensor<{{n}}xf64>) -> tensor<{{n}}xf64> {
  %space = linalg.init_tensor [{{n}}] : tensor<{{n}}xf64>
  %0 = linalg.generic
    {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
    ins(%arg0, %arg1 : tensor<{{n}}xf64>, tensor<{{n}}xf64>)
    outs(%space : tensor<{{n}}xf64>) {
  ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
    %1 = arith.addf %arg2, %arg3 : f64
    linalg.yield %1 : f64
  } -> tensor<{{n}}xf64>
  // %0 = arith.addf %arg0, %arg1 : tensor<{{n}}xf64>
  return %0 : tensor<{{n}}xf64>
}

func @lagrad_vecadd(%arg0: tensor<{{n}}xf64>, %arg1: tensor<{{n}}xf64>, %g: tensor<{{n}}xf64>) -> tensor<{{n}}xf64> {
  %f = constant @vecadd : (tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  %df = standalone.grad %f {of = [0], grad_signal = true} : (tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>, (tensor<{{n}}xf64>, tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  %res = call_indirect %df(%arg0, %arg1, %g) : (tensor<{{n}}xf64>, tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  return %res : tensor<{{n}}xf64>
}

#map0 = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
func @__grad_mvectorscalar(%arg0: tensor<16384xf64>, %arg1: f64, %arg2: tensor<16384xf64>) -> f64 {
  %cst = arith.constant dense<0.000000e+00> : tensor<f64>
  %0 = linalg.generic {indexing_maps = [#map1, #map1, #map0], iterator_types = ["reduction"]} ins(%arg0, %arg2 : tensor<16384xf64>, tensor<16384xf64>) outs(%cst : tensor<f64>) {
  ^bb0(%arg3: f64, %arg4: f64, %arg5: f64):  // no predecessors
    // %2 = arith.mulf %arg4, %arg3 : f64
    %3 = arith.addf %arg4, %arg5 : f64
    linalg.yield %3 : f64
  } -> tensor<f64>
  %1 = tensor.extract %0[] : tensor<f64>
  return %1 : f64
}
func @lagrad_vecscal(%arg0: tensor<16384xf64>, %arg1: f64, %arg2: tensor<16384xf64>) -> f64 {
  %0 = call @__grad_mvectorscalar(%arg0, %arg1, %arg2) : (tensor<16384xf64>, f64, tensor<16384xf64>) -> f64
  return %0 : f64
}

// func @mvectorscalar(%arg0: tensor<{{n}}xf64>, %arg1: f64) -> tensor<{{n}}xf64> {
//   %space = arith.constant dense<0.0> : tensor<{{n}}xf64>
//   %res = linalg.generic
//     {
//       indexing_maps = [#map, #map],
//       iterator_types = ["parallel"]
//     }
//     ins(%arg0 : tensor<{{n}}xf64>)
//     outs(%space : tensor<{{n}}xf64>) {
//   ^bb0(%arg2: f64, %arg3: f64):
//     %0 = arith.mulf %arg2, %arg1 : f64
//     linalg.yield %0 : f64
//   } -> tensor<{{n}}xf64>
//   return %res : tensor<{{n}}xf64>
// }

// func @lagrad_vecscal(%arg0: tensor<{{n}}xf64>, %arg1: f64, %g: tensor<{{n}}xf64>) -> f64 {
//   %f = constant @mvectorscalar : (tensor<{{n}}xf64>, f64) -> tensor<{{n}}xf64>
//   %df = standalone.grad %f {of = [1], grad_signal = true} :
//     (tensor<{{n}}xf64>, f64) -> tensor<{{n}}xf64>,
//     (tensor<{{n}}xf64>, f64, tensor<{{n}}xf64>) -> f64
//   %res = call_indirect %df(%arg0, %arg1, %g) : (tensor<{{n}}xf64>, f64, tensor<{{n}}xf64>) -> f64
//   return %res : f64
// }

func @matmul(%A: tensor<{{n}}x{{n}}xf64>, %B: tensor<{{n}}x{{n}}xf64>) -> tensor<{{n}}x{{n}}xf64> {
  %zero = arith.constant 0.0 : f64
  %space = linalg.init_tensor [{{n}}, {{n}}] : tensor<{{n}}x{{n}}xf64>
  %init = linalg.fill(%zero, %space) : f64, tensor<{{n}}x{{n}}xf64> -> tensor<{{n}}x{{n}}xf64>
  %res = linalg.matmul ins(%A, %B : tensor<{{n}}x{{n}}xf64>, tensor<{{n}}x{{n}}xf64>) outs(%init : tensor<{{n}}x{{n}}xf64>) -> tensor<{{n}}x{{n}}xf64>
  return %res : tensor<{{n}}x{{n}}xf64>
}

func @lagrad_matmul(%A: tensor<{{n}}x{{n}}xf64>, %B: tensor<{{n}}x{{n}}xf64>) -> tensor<{{n}}x{{n}}xf64> {
  %f = constant @matmul : (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}x{{n}}xf64>) -> tensor<{{n}}x{{n}}xf64>
  %df = standalone.grad %f {of = [0]} :
    (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}x{{n}}xf64>) -> tensor<{{n}}x{{n}}xf64>,
    (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}x{{n}}xf64>) -> tensor<{{n}}x{{n}}xf64>
  %res = call_indirect %df(%A, %B) : (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}x{{n}}xf64>) -> tensor<{{n}}x{{n}}xf64>
  return %res : tensor<{{n}}x{{n}}xf64>
}
