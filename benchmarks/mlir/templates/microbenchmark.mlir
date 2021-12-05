// #elementwise_2d = affine_map<(d0, d1) -> (d0, d1)>
// #first_2d = affine_map<(d0, d1) -> (d0)>
// #map0 = affine_map<(d0, d1) -> (d0, d1)>
// #map1 = affine_map<(d0, d1) -> (d0)>
// #map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
// #map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
// #map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// #map5 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// #map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// #map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// #map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
// #map9 = affine_map<(d0, d1) -> (d1)>
// #map10 = affine_map<(d0) -> (d0)>
// #map11 = affine_map<(d0) -> ()>
// #map12 = affine_map<(d0, d1, d2) -> (d0)>
// func @mlir_elementwise_exp(%arg0: tensor<{{k}}x{{d}}xf64>) -> tensor<{{k}}x{{d}}xf64> {
//   %0 = math.exp %arg0 : tensor<{{k}}x{{d}}xf64>
//   return %0 : tensor<{{k}}x{{d}}xf64>
// }

// func @mlir_row_sum(%arg0: tensor<{{k}}x{{d}}xf64>) -> tensor<{{k}}xf64> {
//   %init = arith.constant dense<0.0> : tensor<{{k}}xf64>
//   %1 = linalg.generic
//     {indexing_maps = [#elementwise_2d, #first_2d], iterator_types = ["parallel", "reduction"]}
//     ins(%arg0 : tensor<{{k}}x{{d}}xf64>)
//     outs(%init : tensor<{{k}}xf64>) {
//   ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
//     %43 = arith.addf %arg7, %arg8 : f64
//     linalg.yield %43 : f64
//   } -> tensor<{{k}}xf64>
//   return %1 : tensor<{{k}}xf64>
// }

// func @mlir_broadcasted_sub(%arg0: tensor<{{n}}x{{d}}xf64>, %arg1: tensor<{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64> {
//   %cst_7 = arith.constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf64>
//   %2 = linalg.generic
//     {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]}
//     ins(%arg0, %arg1 : tensor<{{n}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>)
//     outs(%cst_7 : tensor<{{n}}x{{k}}x{{d}}xf64>) {
//   ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
//     %43 = arith.subf %arg7, %arg8 : f64
//     linalg.yield %43 : f64
//   } -> tensor<{{n}}x{{k}}x{{d}}xf64>
//   return %2 : tensor<{{n}}x{{k}}x{{d}}xf64>
// }


// func @mlir_inner_einsum(%arg0: tensor<{{k}}x{{d}}x{{d}}xf64>, %arg1: tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64> {
//   %cst_7 = arith.constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf64>
//   %3 = linalg.generic
//     {
//       indexing_maps = [
//         affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>,
//         affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
//         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//       ],
//       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//     }
//     ins(%arg0, %arg1 : tensor<{{k}}x{{d}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>)
//     outs(%cst_7 : tensor<{{n}}x{{k}}x{{d}}xf64>) {
//   ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
//     %43 = arith.mulf %arg7, %arg8 : f64
//     %44 = arith.addf %43, %arg9 : f64
//     linalg.yield %44 : f64
//   } -> tensor<{{n}}x{{k}}x{{d}}xf64>
//   return %3 : tensor<{{n}}x{{k}}x{{d}}xf64>
// }


// func @mlir_squared_mult(%0: tensor<{{k}}x{{d}}xf64>, %2: tensor<{{n}}x{{k}}x{{d}}xf64>, %3: tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64> {
//   %cst_7 = arith.constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf64>
//   %4 = linalg.generic
//     {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]}
//     ins(%0, %2, %3 : tensor<{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>)
//     outs(%cst_7 : tensor<{{n}}x{{k}}x{{d}}xf64>) {
//   ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
//     %43 = arith.mulf %arg7, %arg8 : f64
//     %44 = arith.addf %43, %arg9 : f64
//     %45 = arith.mulf %44, %44 : f64
//     linalg.yield %45 : f64
//   } -> tensor<{{n}}x{{k}}x{{d}}xf64>
//   return %4 : tensor<{{n}}x{{k}}x{{d}}xf64>
// }

// func @mlir_batched_rowsum(%arg0: tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}xf64> {
//   %cst_6 = arith.constant dense<0.0> : tensor<{{n}}x{{k}}xf64>
//   %5 = linalg.generic
//     {indexing_maps = [#map4, #map8], iterator_types = ["parallel", "parallel", "reduction"]}
//     ins(%arg0 : tensor<{{n}}x{{k}}x{{d}}xf64>)
//     outs(%cst_6 : tensor<{{n}}x{{k}}xf64>) {
//   ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
//     %43 = arith.addf %arg7, %arg8 : f64
//     linalg.yield %43 : f64
//   } -> tensor<{{n}}x{{k}}xf64>
//   return %5 : tensor<{{n}}x{{k}}xf64>
// }

// func @mlir_scalar_mult_sub(%6: tensor<{{k}}xf64>, %5: tensor<{{n}}x{{k}}xf64>) -> tensor<{{n}}x{{k}}xf64> {
//   %cst_5 = arith.constant 0.5 : f64
//   %cst_6 = arith.constant dense<0.0> : tensor<{{n}}x{{k}}xf64>
//   %7 = linalg.generic
//     {indexing_maps = [#map9, #map0, #map0], iterator_types = ["parallel", "parallel"]}
//     ins(%6, %5 : tensor<{{k}}xf64>, tensor<{{n}}x{{k}}xf64>)
//     outs(%cst_6 : tensor<{{n}}x{{k}}xf64>) {
//   ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
//     %43 = arith.mulf %arg8, %cst_5 : f64
//     %44 = arith.subf %arg7, %43 : f64
//     linalg.yield %44 : f64
//   } -> tensor<{{n}}x{{k}}xf64>
//   return %7 : tensor<{{n}}x{{k}}xf64>
// }

func @mlir_matvec(%arg0: memref<{{n}}x{{n}}xf64>, %arg1: memref<{{n}}xf64>, %arg2: memref<{{n}}xf64>) -> memref<{{n}}xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cn = arith.constant {{n}} : index
  scf.for %iv = %c0 to %cn step %c1 {
    scf.for %jv = %c0 to %iv step %c1 {
      %0 = memref.load %arg0[%iv, %jv] : memref<{{n}}x{{n}}xf64>
      %1 = memref.load %arg1[%jv] : memref<{{n}}xf64>
      %2 = memref.load %arg2[%iv] : memref<{{n}}xf64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %3, %2 : f64
      memref.store %4, %arg2[%iv] : memref<{{n}}xf64>
    }
  }
  // linalg.matvec ins(%arg0, %arg1 : memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>) outs(%arg2 : memref<{{n}}xf64>)
  return %arg2 : memref<{{n}}xf64>
}

func @mlir_tri_matvec(%arg0: memref<{{(n * (n - 1) / 2) | round | int}}xf64>, %arg1: memref<{{n}}xf64>, %arg2: memref<{{n}}xf64>) -> memref<{{n}}xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cn = arith.constant {{n}} : index
  scf.for %iv = %c0 to %cn step %c1 iter_args(%arg3 = %c0) -> (index) {
    %6 = scf.for %jv = %c0 to %iv step %c1 iter_args(%arg4 = %arg3) -> (index) {
      %0 = memref.load %arg0[%arg4] : memref<{{(n * (n - 1) / 2) | round | int}}xf64>
      %1 = memref.load %arg1[%jv] : memref<{{n}}xf64>
      %2 = memref.load %arg2[%iv] : memref<{{n}}xf64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %3, %2 : f64
      memref.store %4, %arg2[%iv] : memref<{{n}}xf64>
      %5 = arith.addi %arg4, %c1 : index
      scf.yield %5 : index
    }
    scf.yield %6 : index
  }
  return %arg2 : memref<{{n}}xf64>
}
