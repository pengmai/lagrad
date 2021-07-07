func @dot(%a : tensor<32768xf32>, %b : tensor<32768xf32>) -> tensor<f32> {
  %dummy = constant dense<0.0> : tensor<f32>
  %res = linalg.dot ins(%a, %b : tensor<32768xf32>, tensor<32768xf32>) outs(%dummy : tensor<f32>) -> tensor<f32>
  return %res : tensor<f32>
}

func @ddot(%a : tensor<32768xf32>, %b : tensor<32768xf32>) -> tensor<*xf32> {
  %f = constant @dot : (tensor<32768xf32>, tensor<32768xf32>) -> tensor<f32>
  %df = standalone.grad %f : (tensor<32768xf32>, tensor<32768xf32>) -> tensor<f32>, (tensor<32768xf32>, tensor<32768xf32>) -> tensor<32768xf32>
  %grad = call_indirect %df(%a, %b) : (tensor<32768xf32>, tensor<32768xf32>) -> tensor<32768xf32>
  %casted = tensor.cast %grad : tensor<32768xf32> to tensor<*xf32>
  return %casted : tensor<*xf32>
}

// func @mlir_sum(%a : tensor<32768xf32>) -> f32 {
//   %sum_0 = constant 0.0 : f32
//   %0 = constant 0 : index
//   %10 = constant 10 : index
//   %1 = constant 1 : index
//   %sum = scf.for %iv = %0 to %10 step %1 iter_args(%sum_iter = %sum_0) -> f32 {
//     %t = tensor.extract %a[%iv] : tensor<32768xf32>
//     %sum_next = addf %sum_iter, %t : f32
//     scf.yield %sum_next : f32
//   }
//   return %sum : f32
// }
