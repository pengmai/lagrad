// Comprehensive bufferize erases the return value which is required by the Enzyme transformation pass,
// while regular bufferization doesn't write to the output buffer.
func @tdot(%A: tensor<4xf64>, %B: tensor<4xf64>) -> tensor<f64> {
  %space = arith.constant dense<0.0> : tensor<f64>
  %res = linalg.dot ins(%A, %B : tensor<4xf64>, tensor<4xf64>) outs(%space: tensor<f64>) -> tensor<f64>
  return %res : tensor<f64>
}

// What does this intermediate step need to look like?
// func @desired_after(%A: tensor<4xf64>, %B: tensor<4xf64>, %out: tensor<f64>) -> f64 {
//   %space = arith.constant dense<0.0> : tensor<f64>
//   %res = linalg.dot ins(%A, %B : tensor<4xf64>, tensor<4xf64>) outs(%out: tensor<f64>) -> tensor<f64>
//   %ret = arith.constant 0.0 : f64
//   return %ret : f64
// }

// This is the desired after state.
// Maybe bufferization has to happen in the same step?
// 1. Output tensors are replaced with output arguments
//   - As part of this transformation, the output argument
// 2. A dummy zero return value is added to the end
// func @post_bufferization(%A: memref<4xf64>, %B: memref<4xf64>, %out: memref<f64>) -> f64 {
//   linalg.dot ins(%A, %B : memref<4xf64>, memref<4xf64>) outs(%out: memref<f64>)
//   %ret = arith.constant 0.0 : f64
//   return %ret : f64
// }

func @main() -> i64 {
  %f = constant @tdot : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
  %casted = builtin.unrealized_conversion_cast %f : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64> to (memref<4xf64>, memref<4xf64>) -> memref<f64>
  %df = standalone.diff %casted : (memref<4xf64>, memref<4xf64>) -> memref<f64>, (memref<4xf64>, memref<4xf64>, memref<4xf64>, memref<4xf64>, memref<f64>, memref<f64>) -> f64
  %ret = arith.constant 0 : i64
  return %ret : i64
}
