func @matmul(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>) -> tensor<?x?xf64> {
  %zero = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d1 = tensor.dim %A, %c0 : tensor<?x?xf64>
  %d2 = tensor.dim %B, %c1 : tensor<?x?xf64>
  %space = linalg.init_tensor [%d1, %d2] : tensor<?x?xf64>
  // %space = linalg.init_tensor [?, ?] : tensor<?x?xf64>
  %init = linalg.fill(%zero, %space) : f64, tensor<?x?xf64> -> tensor<?x?xf64>
  %res = linalg.matmul ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>) outs(%init : tensor<?x?xf64>) -> tensor<?x?xf64>
  return %res : tensor<?x?xf64>
}

func @lagrad_matmul(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>) -> tensor<?x?xf64> {
  %f = constant @matmul : (tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
  %df = standalone.grad %f {of = [0]} :
    (tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>,
    (tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
  %res = call_indirect %df(%A, %B) : (tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
  return %res : tensor<?x?xf64>
}
