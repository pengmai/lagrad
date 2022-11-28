func @rowhot_insert(%arg: tensor<10x3xf64> {sparse_tensor.sparsity = "onehot"}) -> tensor<10x4xf64> {
  %space = linalg.init_tensor [10, 4] : tensor<10x4xf64>
  %zero = arith.constant 0.0 : f64
  %dest = linalg.fill(%zero, %space) : f64, tensor<10x4xf64> -> tensor<10x4xf64>
  // %dest = linalg.fill ins(%zero : f64) outs(%space : tensor<10x4xf64>) -> tensor<10x4xf64>
  %res = tensor.insert_slice %arg into %dest[0, 0] [10, 3] [1, 1] : tensor<10x3xf64> into tensor<10x4xf64>
  return %res : tensor<10x4xf64>
}
