// Compute the sum along rows of the matrix.
func.func @rowsum(%arg0: tensor<4x4xf64>) -> tensor<4xf64> {
  %res_init = arith.constant dense<0.0> : tensor<4xf64>
  %res = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%arg0 : tensor<4x4xf64>)
    outs(%res_init : tensor<4xf64>) {
  ^bb0(%arg1: f64, %arg2: f64):
    %0 = arith.addf %arg1, %arg2 : f64
    linalg.yield %0 : f64
  } -> tensor<4xf64>
  return %res : tensor<4xf64>
}

func.func @outer(%arg0: tensor<4x4xf64>) -> tensor<4xf64> {
  %0 = call @rowsum(%arg0) : (tensor<4x4xf64>) -> tensor<4xf64>
  %1 = math.log %0 : tensor<4xf64>
  %2 = call @rowsum(%arg0) : (tensor<4x4xf64>) -> tensor<4xf64>
  %3 = arith.mulf %1, %2 : tensor<4xf64>
  return %3 : tensor<4xf64>
}

func.func private @printMemrefF64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func.func @main() {
  %cst = arith.constant dense<[
    [ 1.,  2.,  3.,  4.],
    [ 5.,  6.,  7.,  8.],
    [ 9., 10., 11., 12.],
    [13., 14., 15., 16.]
  ]> : tensor<4x4xf64>

  %f = constant @outer : (tensor<4x4xf64>) -> tensor<4xf64>
  %df = standalone.grad %f : (tensor<4x4xf64>) -> tensor<4xf64>, (tensor<4x4xf64>) -> tensor<4x4xf64>

  %res = call_indirect %df(%cst) : (tensor<4x4xf64>) -> tensor<4x4xf64>
  %U = tensor.cast %res : tensor<4x4xf64> to tensor<*xf64>

  // %res = call_indirect %f(%cst) : (tensor<4x4xf64>) -> tensor<4xf64>
  // %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  return
}
