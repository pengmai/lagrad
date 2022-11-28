// Minimal working example of a dot product.
func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

func @dot(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<f32> {
  %res = arith.constant dense<0.0> : tensor<f32> // %res is used only for the shape information.
  %val = linalg.dot ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%res : tensor<f32>) -> tensor<f32>
  return %val : tensor<f32>
}

func @main() {
  %A = arith.constant dense<[0.1, 1.0, 2.0, -3.0]> : tensor<4xf32>
  %B = arith.constant dense<[-5.0, 3.4, -10.2, 3.33]> : tensor<4xf32>

  %val = lagrad.grad @dot(%A, %B) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // %df = standalone.grad %f : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>, (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // %val = call_indirect %df(%A, %B) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  %U = tensor.cast %val : tensor<4xf32> to tensor<*xf32>
  call @print_memref_f32(%U) : (tensor<*xf32>) -> ()
  return
}
