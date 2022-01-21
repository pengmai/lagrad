func @sigmoid(%el: f64) -> f64 {
  %negel = arith.negf %el : f64
  %nexp = math.exp %negel : f64
  %one = arith.constant 1.0 : f64
  %denom = arith.addf %nexp, %one : f64
  %res = arith.divf %one, %denom : f64
  return %res : f64
}

#map0 = affine_map<(d0) -> (d0)>
func @withcall(%x: tensor<4xf64>) -> tensor<4xf64> {
  %zero = arith.constant dense<0.0> : tensor<4xf64>
  %z = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %outer_space = linalg.init_tensor [4] : tensor<4xf64>
  %outer_init = linalg.fill(%z, %outer_space) : f64, tensor<4xf64> -> tensor<4xf64>
  %outer_res = scf.for %iv = %c0 to %c3 step %c1 iter_args(%space = %outer_init) -> tensor<4xf64> {
    %res = linalg.generic
      {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
      ins(%x : tensor<4xf64>)
      outs(%zero : tensor<4xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = call @sigmoid(%arg0) : (f64) -> f64
      linalg.yield %0 : f64
    } -> tensor<4xf64>
    %space_next = arith.addf %res, %space : tensor<4xf64>
    scf.yield %space_next : tensor<4xf64>
  }
  return %outer_res : tensor<4xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %A = arith.constant dense<[1., 2., 3., 4.]> : tensor<4xf64>
  %f = constant @withcall : (tensor<4xf64>) -> tensor<4xf64>
  %df = standalone.grad %f : (tensor<4xf64>) -> tensor<4xf64>, (tensor<4xf64>) -> tensor<4xf64>
  %res = call_indirect %df(%A) : (tensor<4xf64>) -> tensor<4xf64>
  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
