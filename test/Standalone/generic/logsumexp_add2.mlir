// This is the version of logsumexp that appears in LSTMs

func.func @mlogsumexp(%t: tensor<4xf64>) -> f64 {
  %out_init = arith.constant dense<0.0> : tensor<f64>
  %lse = linalg.generic
    {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    }
    ins(%t : tensor<4xf64>)
    outs(%out_init : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = math.exp %arg0 : f64
    %1 = arith.addf %0, %arg1 : f64
    linalg.yield %1 : f64
  } -> tensor<f64>
  %lse_v = tensor.extract %lse[] : tensor<f64>
  %two = arith.constant 2.0 : f64
  %lse_2 = arith.addf %lse_v, %two : f64
  %lse_l = math.log %lse_2 : f64
  return %lse_l : f64
}

func.func private @printMemrefF64(tensor<*xf64>) attributes {llvm.emit_c_interface}

func.func @main() {
  %A = arith.constant dense<[1., 2., 3., 4.]> : tensor<4xf64>
  %f = constant @mlogsumexp : (tensor<4xf64>) -> f64
  %df = standalone.grad %f : (tensor<4xf64>) -> f64, (tensor<4xf64>) -> tensor<4xf64>
  %res = call_indirect %df(%A) : (tensor<4xf64>) -> tensor<4xf64>
  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  return
}
