// This is hard to make work properly. Going to leave it for now.
func.func @f(%A: tensor<4x4xf64>, %B: tensor<4xf64>, %C: tensor<4xf64>) -> tensor<4xf64> {
  %out_0 = arith.mulf %C, %B : tensor<4xf64>
  %out_1 = linalg.matvec ins(%A, %B : tensor<4x4xf64>, tensor<4xf64>) outs(%out_0 : tensor<4xf64>) -> tensor<4xf64>
  return %out_1 : tensor<4xf64>
}

func.func private @printMemrefF64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func.func @main() {
  %A = arith.constant dense<[
    [0.377, 0.283, 0.155, 0.858],
    [0.3  , 0.431, 0.851, 0.137],
    [0.776, 0.555, 0.771, 0.233],
    [0.623, 0.193, 0.005, 0.691]
  ]> : tensor<4x4xf64>
  %B = arith.constant dense<[4.0, -3.0,  2.0,  1.0]> : tensor<4xf64>
  %C = arith.constant dense<[1.0,  0.2, -2.3, -1.7]> : tensor<4xf64>
  %res = lagrad.grad @f(%A, %B, %C) {of = [1]} : (tensor<4x4xf64>, tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  return
}
