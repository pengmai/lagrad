func.func @slice_add_nocollapse(%arg0: tensor<2x2xf64>) -> tensor<f64> {
  %0 = tensor.extract_slice %arg0[0, 1] [2, 1] [1, 1] : tensor<2x2xf64> to tensor<2x1xf64>
  %1 = tensor.extract_slice %arg0[0, 0] [2, 1] [1, 1] : tensor<2x2xf64> to tensor<2x1xf64>
  %cst = arith.constant dense<0.0> : tensor<2x1xf64>
  %cst_1 = arith.constant 2.0 : f64
  %3 = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%0 : tensor<2x1xf64>)
    outs(%cst : tensor<2x1xf64>) {
  ^bb0(%arg1: f64, %arg2: f64):
    %7 = arith.mulf %arg1, %cst_1 : f64
    linalg.yield %7 : f64
  } -> tensor<2x1xf64>
  %5 = arith.addf %3, %1 : tensor<2x1xf64>
  %cst_2 = arith.constant dense<0.0> : tensor<f64>
  %6 = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>],
      iterator_types = ["reduction", "reduction"]
    }
    ins(%5 : tensor<2x1xf64>)
    outs(%cst_2 : tensor<f64>) {
  ^bb0(%arg1: f64, %arg2: f64):
    %7 = arith.addf %arg1, %arg2 : f64
    linalg.yield %7 : f64
  } -> tensor<f64>
  return %6 : tensor<f64>
}

func.func private @printMemrefF64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func.func @main() {
  %f = constant @slice_add_nocollapse : (tensor<2x2xf64>) -> tensor<f64>
  %df = standalone.grad %f : (tensor<2x2xf64>) -> tensor<f64>, (tensor<2x2xf64>) -> tensor<2x2xf64>
  %arg = arith.constant dense<[[3.4, 2.3], [2.1, -3.2]]> : tensor<2x2xf64>
  %res = call_indirect %df(%arg) : (tensor<2x2xf64>) -> tensor<2x2xf64>
  // %res = call @__grad_slice_add_nocollapse(%arg) : (tensor<2x2xf64>) -> tensor<2x2xf64>
  %U = tensor.cast %res : tensor<2x2xf64> to tensor<*xf64>
  call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  return
}
