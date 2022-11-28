func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
func @logsumexp(%A : tensor<4xf64>) -> f64 {
  %max_space = linalg.init_tensor [] : tensor<f64>
  %c0 = arith.constant 0 : index
  %max_init_val = tensor.extract %A[%c0] : tensor<4xf64>
  %max_init = tensor.insert %max_init_val into %max_space[] : tensor<f64>

  %max_t = linalg.generic
    {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]}
    ins(%A : tensor<4xf64>)
    outs(%max_init : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %p = arith.cmpf "ogt", %arg0, %arg1 : f64
    %next = scf.if %p -> (f64) {
      scf.yield %arg0 : f64
    } else {
      scf.yield %arg1 : f64
    }
    linalg.yield %next : f64
  } -> tensor<f64>

  %max = tensor.extract %max_t[] : tensor<f64>
  %zero = arith.constant dense<0.0> : tensor<f64>
  %se_noadd_t = linalg.generic
    {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]}
    ins(%A : tensor<4xf64>)
    outs(%zero : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = arith.subf %arg0, %max : f64
    %1 = math.exp %0 : f64
    %2 = arith.addf %1, %arg1 : f64
    linalg.yield %2 : f64
  } -> tensor<f64>
  %se_noadd = tensor.extract %se_noadd_t[] : tensor<f64>
  %lse_noadd = math.log %se_noadd : f64
  %lse = arith.addf %lse_noadd, %max : f64
  return %lse : f64
}

func @main() {
  %A = arith.constant dense<[3.4, -5.5, -1.1, 0.1]> : tensor<4xf64>
  %res = lagrad.grad @logsumexp(%A) : (tensor<4xf64>) -> tensor<4xf64>
  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
