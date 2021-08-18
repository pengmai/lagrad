func @dot(%a : tensor<131072xf32>, %b : tensor<131072xf32>) -> tensor<f32> {
  %dummy = constant dense<0.0> : tensor<f32>
  %res = linalg.dot ins(%a, %b : tensor<131072xf32>, tensor<131072xf32>) outs(%dummy : tensor<f32>) -> tensor<f32>
  return %res : tensor<f32>
}

// // Only take the gradient of the first argument.
// func @ddot(%a : tensor<131072xf32>, %b : tensor<131072xf32>) -> tensor<131072xf32> {
//   %f = constant @dot : (tensor<131072xf32>, tensor<131072xf32>) -> tensor<f32>
//   %df = standalone.grad %f {of=[0]} : (tensor<131072xf32>, tensor<131072xf32>) -> tensor<f32>, (tensor<131072xf32>, tensor<131072xf32>) -> tensor<131072xf32>
//   %grad = call_indirect %df(%a, %b) : (tensor<131072xf32>, tensor<131072xf32>) -> tensor<131072xf32>
//   return %grad : tensor<131072xf32>
// }

func @ddot(%a : tensor<131072xf32>, %b : tensor<131072xf32>) -> (tensor<131072xf32>, tensor<131072xf32>) {
  %f = constant @dot : (tensor<131072xf32>, tensor<131072xf32>) -> tensor<f32>
  %df = standalone.grad %f {of=[0, 1]} : (tensor<131072xf32>, tensor<131072xf32>) -> tensor<f32>, (tensor<131072xf32>, tensor<131072xf32>) -> (tensor<131072xf32>, tensor<131072xf32>)
  %grad:2 = call_indirect %df(%a, %b) : (tensor<131072xf32>, tensor<131072xf32>) -> (tensor<131072xf32>, tensor<131072xf32>)
  return %grad#0, %grad#1 : tensor<131072xf32>, tensor<131072xf32>
}
