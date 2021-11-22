func private @mymul(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
  %0 = arith.mulf %arg0, %arg1 : tensor<3xf64>
  return %0 : tensor<3xf64>
}

func private @myfunc(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
  %zero = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %el = tensor.extract %arg0[%c0] : tensor<3xf64>
  %p = arith.cmpf "oeq", %el, %zero : f64
  %arg1el_space = linalg.init_tensor [3] : tensor<3xf64>
  %final = scf.if %p -> tensor<3xf64> {
    %sinel = math.sin %el : f64
    %arg1el = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg1 : tensor<3xf64>) outs(%arg1el_space : tensor<3xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):
      %0 = arith.mulf %arg2, %sinel : f64
      linalg.yield %0 : f64
    } -> tensor<3xf64>
    %res = call @mymul(%arg0, %arg1el) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    scf.yield %res : tensor<3xf64>
  } else {
    scf.yield %arg0 : tensor<3xf64>
  }
  return %final : tensor<3xf64>
}

func @main() {
  %A = arith.constant dense<[1., 2., 3.]> : tensor<3xf64>
  %B = arith.constant dense<[4., 5., 6.]> : tensor<3xf64>

  %f = constant @myfunc : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  %df = standalone.grad %f {of = [0]} : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>, (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  call_indirect %df(%A, %B) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  return
}
