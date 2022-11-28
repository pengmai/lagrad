#map = affine_map<(d0) -> (d0)>

func private @msigmoid(%x: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %nx = arith.negf %x : f64
  %exp = math.exp %nx : f64
  %denom = arith.addf %one, %exp : f64
  %frac = arith.divf %one, %denom : f64
  return %frac : f64
}

func @lstm_model(%weight: tensor<4x14xf64>, %bias: tensor<4x14xf64>, %hidden: tensor<14xf64>, %cell: tensor<14xf64>, %input: tensor<14xf64>) -> tensor<14xf64> {
  %zero_b = arith.constant dense<0.0> : tensor<14xf64>
  %fweight = tensor.extract_slice %weight[0, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
  %iweight = tensor.extract_slice %weight[1, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
  %oweight = tensor.extract_slice %weight[2, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
  %cweight = tensor.extract_slice %weight[3, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
  %fbias   = tensor.extract_slice %bias  [0, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
  %ibias   = tensor.extract_slice %bias  [1, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
  %obias   = tensor.extract_slice %bias  [2, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
  %cbias   = tensor.extract_slice %bias  [3, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
  %forget = linalg.generic
    {
      indexing_maps = [#map, #map, #map, #map],
      iterator_types = ["parallel"]
    }
    ins(%input, %fweight, %fbias : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>)
    outs(%zero_b : tensor<14xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    %2 = call @msigmoid(%1) : (f64) -> f64
    linalg.yield %2 : f64
  } -> tensor<14xf64>
  %ingate = linalg.generic
    {
      indexing_maps = [#map, #map, #map, #map],
      iterator_types = ["parallel"]
    }
    ins(%hidden, %iweight, %ibias : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>)
    outs(%zero_b : tensor<14xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    %2 = call @msigmoid(%1) : (f64) -> f64
    linalg.yield %2 : f64
  } -> tensor<14xf64>
  %outgate = linalg.generic
    {
      indexing_maps = [#map, #map, #map, #map],
      iterator_types = ["parallel"]
    }
    ins(%input, %oweight, %obias : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>)
    outs(%zero_b : tensor<14xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    %2 = call @msigmoid(%1) : (f64) -> f64
    linalg.yield %2 : f64
  } -> tensor<14xf64>
  %change = linalg.generic
    {
      indexing_maps = [#map, #map, #map, #map],
      iterator_types = ["parallel"]
    }
    ins(%hidden, %cweight, %cbias : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>)
    outs(%zero_b : tensor<14xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    %2 = math.tanh %1 : f64
    linalg.yield %2 : f64
  } -> tensor<14xf64>

  %cell_new = linalg.generic
    {
      indexing_maps = [#map, #map, #map, #map, #map],
      iterator_types = ["parallel"]
    }
    ins(
      %cell,
      %forget,
      %ingate,
      %change : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>
    )
    outs(%zero_b : tensor<14xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.mulf %arg2, %arg3 : f64
    %2 = arith.addf %0, %1 : f64
    linalg.yield %2 : f64
  } -> tensor<14xf64>

  %hidden_next = linalg.generic
    {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel"]
    }
    ins(%outgate, %cell_new : tensor<14xf64>, tensor<14xf64>)
    outs(%zero_b : tensor<14xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = math.tanh %arg1 : f64
    %1 = arith.mulf %arg0, %0 : f64
    linalg.yield %1 : f64
  } -> tensor<14xf64>

  return %hidden_next : tensor<14xf64>
}

func @lagrad_lstm_model(
  %weight: tensor<4x14xf64>,
  %bias: tensor<4x14xf64>,
  %hidden: tensor<14xf64>,
  %cell: tensor<14xf64>,
  %input: tensor<14xf64>
) -> (
  tensor<4x14xf64>,
  tensor<4x14xf64>,
  tensor<14xf64>,
  tensor<14xf64>,
  tensor<14xf64>
) {
  %res:5 = lagrad.grad @lstm_model(%weight, %bias, %hidden, %cell, %input) {of = [0, 1, 2, 3, 4]} : (
    tensor<4x14xf64>,
    tensor<4x14xf64>,
    tensor<14xf64>,
    tensor<14xf64>,
    tensor<14xf64>
  ) -> (
    tensor<4x14xf64>,
    tensor<4x14xf64>,
    tensor<14xf64>,
    tensor<14xf64>,
    tensor<14xf64>
  )
  return %res#0, %res#1, %res#2, %res#3, %res#4 : tensor<4x14xf64>, tensor<4x14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>
}

// func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

// func @main() {
//   %weight = arith.constant dense<[
//     [8.8266e-01, 6.7157e-01, 3.9910e-02, 1.5030e-01, 2.4490e-01, 1.8165e-01, 6.0816e-01, 6.4715e-01, 8.3273e-01, 1.6309e-01, 4.6190e-01, 9.4787e-02, 5.6751e-01, 4.3819e-01],
//     [4.7996e-01, 3.3645e-01, 3.5138e-01, 9.0498e-01, 3.7815e-01, 9.8501e-01, 5.7715e-01, 9.4654e-01, 6.8229e-01, 6.4366e-01, 6.8837e-01, 5.1466e-01, 8.2005e-01, 9.4799e-01],
//     [3.4126e-01, 6.7523e-01, 3.5501e-01, 9.8691e-01, 1.3191e-02, 1.8885e-01, 1.8045e-01, 2.1814e-01, 1.6727e-01, 9.7647e-01, 3.5158e-01, 6.9281e-01, 8.6443e-01, 6.6944e-01],
//     [8.2831e-01, 7.3193e-01, 3.8494e-01, 9.9825e-01, 6.9272e-01, 1.3344e-01, 1.2178e-04, 6.9670e-01, 8.8730e-01, 2.8278e-01, 2.2379e-04, 3.1818e-01, 5.6636e-01, 7.9017e-01]
//   ]> : tensor<4x14xf64>
//   %bias = arith.constant dense<[
//     [6.8976e-01, 9.0010e-01, 5.1352e-01, 5.6847e-01, 9.2742e-01, 6.8320e-01, 7.8179e-01, 8.2767e-01, 2.9498e-01, 9.5646e-01, 9.1959e-01, 1.7504e-01, 2.4087e-01, 3.3704e-01],
//     [2.1885e-01, 1.0908e-01, 7.8185e-01, 3.6932e-01, 6.2499e-01, 3.4247e-02, 1.0544e-01, 8.7869e-01, 6.1999e-01, 8.8827e-01, 3.9653e-01, 5.3414e-01, 5.3185e-01, 4.5053e-01],
//     [7.3838e-01, 7.8870e-03, 9.3418e-01, 1.6877e-01, 9.6079e-01, 6.1498e-01, 2.9224e-01, 3.9197e-01, 5.9654e-01, 1.0579e-01, 1.3294e-03, 2.9002e-01, 8.5928e-01, 9.8352e-01],
//     [9.4208e-01, 5.5315e-01, 8.4616e-01, 1.8816e-01, 8.4349e-02, 3.4637e-02, 3.1831e-01, 9.3476e-02, 3.5207e-01, 5.1579e-01, 9.9954e-01, 3.4240e-01, 1.8631e-01, 1.3253e-02]
//   ]> : tensor<4x14xf64>
//   %hidden = arith.constant dense<[
//     6.6659e-01, 8.0407e-01, 3.5582e-01, 3.8362e-02, 1.0859e-01, 1.7703e-01, 3.6881e-01, 3.8816e-01, 4.2066e-01,
//     1.2587e-02, 5.5262e-01, 7.3115e-01, 4.9832e-01, 4.5906e-01
//   ]> : tensor<14xf64>
//   %cell = arith.constant dense<[
//     5.1489e-01, 9.3151e-01, 5.0919e-01, 2.0063e-01, 2.7195e-01, 7.1018e-01, 6.1621e-01, 3.5843e-01, 4.3950e-01,
//     2.2892e-01, 5.1842e-01, 7.3015e-01, 3.3122e-01, 5.1164e-01
//   ]> : tensor<14xf64>
//   %input = arith.constant dense<[0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.]> : tensor<14xf64>

//   %f = constant @lstm_model : (tensor<4x14xf64>, tensor<4x14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) -> tensor<14xf64>
//   %df = standalone.grad %f {of = [0, 1, 2, 3, 4]} : (
//     tensor<4x14xf64>,
//     tensor<4x14xf64>,
//     tensor<14xf64>,
//     tensor<14xf64>,
//     tensor<14xf64>
//   ) -> tensor<14xf64>, (
//     tensor<4x14xf64>,
//     tensor<4x14xf64>,
//     tensor<14xf64>,
//     tensor<14xf64>,
//     tensor<14xf64>
//   ) -> (
//     tensor<4x14xf64>,
//     tensor<4x14xf64>,
//     tensor<14xf64>,
//     tensor<14xf64>,
//     tensor<14xf64>
//   )
//   %res:5 = call_indirect %df(%weight, %bias, %hidden, %cell, %input) : (
//     tensor<4x14xf64>,
//     tensor<4x14xf64>,
//     tensor<14xf64>,
//     tensor<14xf64>,
//     tensor<14xf64>
//   ) -> (
//     tensor<4x14xf64>,
//     tensor<4x14xf64>,
//     tensor<14xf64>,
//     tensor<14xf64>,
//     tensor<14xf64>
//   )
//   %U = tensor.cast %res#0 : tensor<4x14xf64> to tensor<*xf64>
//   call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
//   %U1 = tensor.cast %res#1 : tensor<4x14xf64> to tensor<*xf64>
//   call @print_memref_f64(%U1) : (tensor<*xf64>) -> ()
//   %U2 = tensor.cast %res#2 : tensor<14xf64> to tensor<*xf64>
//   call @print_memref_f64(%U2) : (tensor<*xf64>) -> ()
//   %U3 = tensor.cast %res#3 : tensor<14xf64> to tensor<*xf64>
//   call @print_memref_f64(%U3) : (tensor<*xf64>) -> ()
//   %U4 = tensor.cast %res#4 : tensor<14xf64> to tensor<*xf64>
//   call @print_memref_f64(%U4) : (tensor<*xf64>) -> ()
//   return
// }
