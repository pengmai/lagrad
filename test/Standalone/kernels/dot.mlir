func @dot(%a : tensor<131072xf32>, %b : tensor<131072xf32>) -> tensor<f32> {
  %dummy = constant dense<0.0> : tensor<f32>
  %res = linalg.dot ins(%a, %b : tensor<131072xf32>, tensor<131072xf32>) outs(%dummy : tensor<f32>) -> tensor<f32>
  return %res : tensor<f32>
}

// func @__grad_dot(%arg0: tensor<131072xf32>, %arg1: tensor<131072xf32>) -> tensor<131072xf32> {
//   %cst = constant dense<0.000000e+00> : tensor<131072xf32>
//   %cst_0 = constant 1.000000e+00 : f32

//   %c0 = constant 0 : index
//   // %c1 = constant 1 : index
//   %c131072 = constant 131072 : index

//   // %0 = alloca() : memref<131072xf32>
//   // linalg.fill(%0, %cst_0) : memref<131072xf32>, f32
//   // %1 = tensor_load %0 : memref<131072xf32>
//   %1 = constant dense<1.0> : tensor<131072xf32>

//   %2 = mulf %1, %arg1 : tensor<131072xf32>
//   // %mulbuf = alloc() : memref<131072xf32>
//   // affine.for %iv = %c0 to %c131072 {
//   //   %e1 = tensor.extract %1[%iv] : tensor<131072xf32>
//   //   %e2 = tensor.extract %arg1[%iv] : tensor<131072xf32>
//   //   %mulres = mulf %e1, %e2 : f32
//   //   affine.store %mulres, %mulbuf[%iv] : memref<131072xf32>
//   // }
//   // %2 = tensor_load %mulbuf : memref<131072xf32>

//   %3 = addf %cst, %2 : tensor<131072xf32>
//   // %addbuf = alloc() : memref<131072xf32>
//   // affine.for %iv = %c0 to %c131072 {
//   //   %e1 = tensor.extract %cst[%iv] : tensor<131072xf32>
//   //   %e2 = affine.load %mulbuf[%iv] : memref<131072xf32>
//   //   %addres = addf %e1, %e2 : f32
//   //   store %addres, %addbuf[%iv] : memref<131072xf32>
//   // }
//   // %3 = tensor_load %addbuf : memref<131072xf32>

//   // %4 = alloc() : memref<131072xf32>
//   // linalg.fill(%4, %cst_0) : memref<131072xf32>, f32
//   return %3 : tensor<131072xf32>
// }

// Dot product gradient implemented by hand.
// func @handdot(%a : tensor<131072xf32>, %b : tensor<131072xf32>) -> tensor<131072xf32> {
//   %zero = constant dense<0.0> : tensor<131072xf32>
//   %res = addf %zero, %b : tensor<131072xf32>
//   %res2 = addf %zero, %a : tensor<131072xf32>
//   return %res : tensor<131072xf32>
// }

func @ddot(%a : tensor<131072xf32>, %b : tensor<131072xf32>) -> tensor<*xf32> {
  %f = constant @dot : (tensor<131072xf32>, tensor<131072xf32>) -> tensor<f32>
  %df = standalone.grad %f : (tensor<131072xf32>, tensor<131072xf32>) -> tensor<f32>, (tensor<131072xf32>, tensor<131072xf32>) -> tensor<131072xf32>
  %grad = call_indirect %df(%a, %b) : (tensor<131072xf32>, tensor<131072xf32>) -> tensor<131072xf32>
  // %grad = call @__grad_dot(%a, %b) : (tensor<131072xf32>, tensor<131072xf32>) -> tensor<131072xf32>
  %casted = tensor.cast %grad : tensor<131072xf32> to tensor<*xf32>
  return %casted : tensor<*xf32>
}
