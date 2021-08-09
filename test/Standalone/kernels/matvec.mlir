func @matvec(%M : tensor<512x1024xf32>, %x : tensor<1024xf32>) -> tensor<512xf32> {
  %dummy = constant dense<0.0> : tensor<512xf32>
  %res = linalg.matvec ins(%M, %x : tensor<512x1024xf32>, tensor<1024xf32>) outs(%dummy : tensor<512xf32>) -> tensor<512xf32>
  return %res : tensor<512xf32>
}

// #map0 = affine_map<(d0, d1) -> (d0)>
// #map1 = affine_map<(d0, d1) -> (d1)>
// #map2 = affine_map<(d0, d1) -> (d0, d1)>

// func @__grad_matvec(%arg0: tensor<512x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<512x1024xf32> {
//   %cst_0 = constant dense<1.000000e+00> : tensor<512xf32>
//   %cst_1 = constant dense<0.000000e+00> : tensor<512x1024xf32>

//   // *** Explicit loop (able to optimize the tensor.extract) *** //
//   // %res = alloc() : memref<512x1024xf32>
//   // %c1024 = constant 1024 : index
//   // affine.for %i = 0 to 512 {
//   //   affine.for %j = 0 to 1024 {
//   //     %0 = tensor.extract %cst_0[%i] : tensor<512xf32>
//   //     %1 = tensor.extract %arg1[%j] : tensor<1024xf32>
//   //     // %5 = mulf %0, %1 : f32
//   //     affine.store %1, %res[%i, %j] : memref<512x1024xf32>
//   //   }
//   // }
//   // %loaded = tensor_load %res : memref<512x1024xf32>
//   // return %loaded : tensor<512x1024xf32>
//   // %shape_cst = constant dense<[512, 1024]> : tensor<2xi32>
//   // %shape = tensor_to_memref %shape_cst : memref<2xi32>
//   // %6 = memref_cast %res : memref<524288xf32> to memref<*xf32>
//   // %casted = memref_reshape %6(%shape) : (memref<*xf32>, memref<2xi32>) -> memref<512x1024xf32>
//   // %loaded = tensor_load %casted : memref<512x1024xf32>
//   // return %loaded : tensor<512x1024xf32>

//   // *** linalg.generic using tensors *** //
//   %0 = linalg.generic
//     {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel"]}
//     ins(%cst_0, %arg1 : tensor<512xf32>, tensor<1024xf32>)
//     outs(%cst_1 : tensor<512x1024xf32>) {
//   ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
//     %0 = mulf %arg2, %arg3 : f32
//     // %1 = addf %0, %arg4 : f32
//     linalg.yield %0 : f32
//   } -> tensor<512x1024xf32>
//   // %1 = addf %0, %cst_1 : tensor<512x1024xf32>
//   return %0 : tensor<512x1024xf32>

//   // This is the second argument
//   // %cst_2 = constant dense<0.000000e+00> : tensor<1024xf32>
//   // %5 = linalg.generic
//   //   {indexing_maps = [#map0, #map2, #map1], iterator_types = ["reduction", "parallel"]}
//   //   ins(%cst_0, %arg0 : tensor<512xf32>, tensor<512x1024xf32>)
//   //   outs(%cst_2 : tensor<1024xf32>) {
//   // ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
//   //   %7 = mulf %arg2, %arg3 : f32
//   //   %8 = addf %7, %arg4 : f32
//   //   linalg.yield %8 : f32
//   // } -> tensor<1024xf32>
//   // return %0, %5 : tensor<512x1024xf32>, tensor<1024xf32>
// }

// Take the gradient only of the first argument.
// func @grad_matvec(%M : tensor<512x1024xf32>, %x : tensor<1024xf32>) -> tensor<512x1024xf32> {
//   %f = constant @matvec : (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512xf32>
//   %df = standalone.grad %f {of=[0]}: (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512xf32>, (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512x1024xf32>
//   %res = call_indirect %df(%M, %x) : (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512x1024xf32>
//   // %res = call @__grad_matvec(%M, %x) : (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512x1024xf32>
//   return %res : tensor<512x1024xf32>
// }

// global_memref "private" constant @__constant_1024xf32 : memref<1024xf32> = dense<0.000000e+00>
// global_memref "private" constant @__constant_512xf32_0 : memref<512xf32> = dense<1.000000e+00>
// global_memref "private" constant @__constant_512xf32 : memref<512xf32> = dense<0.000000e+00>
// func @__grad_matvec(%arg0: memref<512x1024xf32>, %arg1: memref<1024xf32>) -> memref<1024xf32> {
//   // %0 = get_global_memref @__constant_512xf32_0 : memref<512xf32>
//   %1 = get_global_memref @__constant_1024xf32 : memref<1024xf32>
//   %2 = alloc() : memref<1024xf32>
//   %cst0 = constant 0.0 : f32
//   affine.for %arg2 = 0 to 1024 {
//     %3 = affine.load %1[%arg2] : memref<1024xf32>
//     affine.store %3, %2[%arg2] : memref<1024xf32>
//   }

//   affine.for %arg3 = 0 to 512 {
//     affine.for %arg2 = 0 to 1024 {
//       // %4 = affine.load %0[%arg3] : memref<512xf32>
//       %5 = affine.load %arg0[%arg3, %arg2] : memref<512x1024xf32>
//       %6 = affine.load %2[%arg2] : memref<1024xf32>
//       // %7 = mulf %4, %5 : f32
//       %8 = addf %5, %6 : f32
//       affine.store %8, %2[%arg2] : memref<1024xf32>
//     }
//   }
//   return %2 : memref<1024xf32>
// }
// func @grad_matvec(%arg0: memref<512x1024xf32>, %arg1: memref<1024xf32>) -> memref<1024xf32> {
//   %0 = call @__grad_matvec(%arg0, %arg1) : (memref<512x1024xf32>, memref<1024xf32>) -> memref<1024xf32>
//   return %0 : memref<1024xf32>
// }

// Take the gradient of only the second argument.
func @grad_matvec(%M : tensor<512x1024xf32>, %x : tensor<1024xf32>) -> tensor<1024xf32> {
  %f = constant @matvec : (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512xf32>
  %df = standalone.grad %f {of=[1]}: (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512xf32>, (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
  %res = call_indirect %df(%M, %x) : (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
  // %res = call @__grad_matvec(%M, %x) : (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
  return %res : tensor<1024xf32>
}

// Take the gradient of both arguments
// func @_grad_matvec(%M : tensor<512x1024xf32>, %x : tensor<1024xf32>) -> (tensor<512x1024xf32>, tensor<1024xf32>) {
//   %f = constant @matvec : (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512xf32>
//   %df = standalone.grad %f {of=[0]}: (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512xf32>, (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512x1024xf32>
//   %res = call_indirect %df(%M, %x) : (tensor<512x1024xf32>, tensor<1024xf32>) -> tensor<512x1024xf32>
//   // %res:2 = call @__grad_matvec(%M, %x) : (tensor<512x1024xf32>, tensor<1024xf32>) -> (tensor<512x1024xf32>, tensor<1024xf32>)
//   return %res#0, %res#1 : tensor<512x1024xf32>, tensor<1024xf32>
// }
