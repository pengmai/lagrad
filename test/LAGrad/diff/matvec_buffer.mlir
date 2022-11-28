// // An elementwise gradient of a matrix-vector multiplication.
// func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

// func @matvec(%arg0: tensor<3x4xf32>, %arg1: tensor<4xf32>, %out: tensor<3xf32>) -> f32 {
//   %val = linalg.matvec ins(%arg0, %arg1 : tensor<3x4xf32>, tensor<4xf32>) outs(%out : tensor<3xf32>) -> tensor<3xf32>
//   // Need to write to the output somehow
//   %mval = memref.buffer_cast %val : memref<3xf32>
//   %mout = memref.buffer_cast %out : memref<3xf32>
//   linalg.copy(%mval, %mout) : memref<3xf32>, memref<3xf32>

//   %ret = arith.constant 0.0 : f32
//   return %ret : f32
// }

// func @main() {
//   %zero = arith.constant 0.0 : f32
//   %M = arith.constant dense<[
//     [0.8,  0.9,  1.2, -4.3],
//     [2.3, -1.1, -7.5, -3.2],
//     [1.2,  1.1,  1.0,  0.0]
//   ]> : tensor<3x4xf32>

//   %dM_init = linalg.init_tensor [3, 4] : tensor<3x4xf32>
//   %dM = linalg.fill(%zero, %dM_init) : f32, tensor<3x4xf32> -> tensor<3x4xf32>

//   %x = arith.constant dense<[-1.2, -1.3, 1.5, 2.2]> : tensor<4xf32>

//   %dx_init = linalg.init_tensor [4] : tensor<4xf32>
//   %dx = linalg.fill(%zero, %dx_init) : f32, tensor<4xf32> -> tensor<4xf32>

//   %out = arith.constant dense<0.0> : tensor<3xf32>
//   %dout = arith.constant dense<1.0> : tensor<3xf32>

//   // %f = constant @matvec : (tensor<3x4xf32>, tensor<4xf32>, tensor<3xf32>) -> f32
//   // %df = lagrad.diff %f : (tensor<3x4xf32>, tensor<4xf32>, tensor<3xf32>) -> f32, (tensor<3x4xf32>, tensor<3x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<3xf32>, tensor<3xf32>) -> f32
//   // call_indirect %df(%M, %dM, %x, %dx, %out, %dout) : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<3xf32>, tensor<3xf32>) -> f32

//   %casted = tensor.cast %dM : tensor<3x4xf32> to tensor<*xf32>
//   call @print_memref_f32(%casted) : (tensor<*xf32>) -> ()
//   return
// }

memref.global "private" constant @__constant_3xf32_0 : memref<3xf32> = dense<1.000000e+00>
memref.global "private" constant @__constant_3xf32 : memref<3xf32> = dense<0.000000e+00>
memref.global "private" constant @__constant_4xf32 : memref<4xf32> = dense<[-1.200000e+00, -1.300000e+00, 1.500000e+00, 2.200000e+00]>
memref.global "private" constant @__constant_3x4xf32 : memref<3x4xf32> = dense<[[8.000000e-01, 0.899999976, 1.200000e+00, -4.300000e+00], [2.300000e+00, -1.100000e+00, -7.500000e+00, -3.200000e+00], [1.200000e+00, 1.100000e+00, 1.000000e+00, 0.000000e+00]]>

func private @print_memref_f32(memref<*xf32>) attributes {llvm.emit_c_interface}

func @matvec(%arg0: memref<3x4xf32>, %arg1: memref<4xf32>, %arg2: memref<3xf32>) -> f32 {
  %0 = memref.alloc() : memref<3xf32>
  linalg.copy(%arg2, %0) : memref<3xf32>, memref<3xf32>
  linalg.matvec ins(%arg0, %arg1 : memref<3x4xf32>, memref<4xf32>) outs(%0 : memref<3xf32>)
  linalg.copy(%0, %arg2) : memref<3xf32>, memref<3xf32>
  %cst = arith.constant 0.000000e+00 : f32
  return %cst : f32
}

func @main() -> i64 {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.get_global @__constant_3x4xf32 : memref<3x4xf32>
  %1 = memref.alloc() : memref<3x4xf32>
  linalg.fill(%cst, %1) : f32, memref<3x4xf32>
  %2 = memref.get_global @__constant_4xf32 : memref<4xf32>
  %3 = memref.alloc() : memref<4xf32>
  linalg.fill(%cst, %3) : f32, memref<4xf32>
  %4 = memref.get_global @__constant_3xf32 : memref<3xf32>
  %5 = memref.get_global @__constant_3xf32_0 : memref<3xf32>

  %f = constant @matvec : (memref<3x4xf32>, memref<4xf32>, memref<3xf32>) -> f32
  %df = lagrad.diff %f : (memref<3x4xf32>, memref<4xf32>, memref<3xf32>) -> f32, (memref<3x4xf32>, memref<3x4xf32>, memref<4xf32>, memref<4xf32>, memref<3xf32>, memref<3xf32>) -> f32
  call_indirect %df(%0, %1, %2, %3, %4, %5) : (memref<3x4xf32>, memref<3x4xf32>, memref<4xf32>, memref<4xf32>, memref<3xf32>, memref<3xf32>) -> f32
  %6 = memref.cast %1 : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%6) : (memref<*xf32>) -> ()

  %ret = arith.constant 0 : i64
  return %ret : i64
}
