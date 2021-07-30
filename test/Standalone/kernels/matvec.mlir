func @matvec(%M : tensor<3x4xf32>, %x : tensor<4xf32>) -> tensor<3xf32> {
  %dummy = constant dense<0.0> : tensor<3xf32>
  %res = linalg.matvec ins(%M, %x : tensor<3x4xf32>, tensor<4xf32>) outs(%dummy : tensor<3xf32>) -> tensor<3xf32>
  return %res : tensor<3xf32>
}

// func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @grad_matvec(%M : tensor<3x4xf32>, %x : tensor<4xf32>) -> tensor<*xf32> {
  %f = constant @matvec : (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3xf32>
  %df = standalone.grad %f : (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3xf32>, (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  %res = call_indirect %df(%M, %x) : (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  %casted = tensor.cast %res : tensor<3x4xf32> to tensor<*xf32>
  return %casted : tensor<*xf32>
}

// func @main() {
//   %M = constant dense<1.2> : tensor<3x4xf32>
//   %x = constant dense<1.1> : tensor<4xf32>
// }

// --- Reference ---
