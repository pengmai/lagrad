#map = affine_map<(d0) -> (d0)>

func @msigmoid(%x: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %nx = arith.negf %x : f64
  %exp = math.exp %nx : f64
  %denom = arith.addf %one, %exp : f64
  %frac = arith.divf %one, %denom : f64
  return %frac : f64
}

func @lstm_predict(%main_params: tensor<2x2x4x14xf64>, %extra_params: tensor<3x14xf64>, %state_outer: tensor<2x2x14xf64>, %x: tensor<14xf64>) -> tensor<14xf64> {
  %w2 = tensor.extract_slice %extra_params[0, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
  %x2 = arith.mulf %x, %w2 : tensor<14xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cl = arith.constant 2 : index
  %cb = arith.constant 14 : index
  %zero_b = arith.constant dense<0.0> : tensor<14xf64>

  %xp:2 = scf.for %iv = %c0 to %cl step %c1 iter_args(%input = %x2, %state = %state_outer) -> (tensor<14xf64>, tensor<2x2x14xf64>) {
    %fweight = tensor.extract_slice %main_params[%iv, 0, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
    %fbias   = tensor.extract_slice %main_params[%iv, 1, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
    %iweight = tensor.extract_slice %main_params[%iv, 0, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
    %ibias   = tensor.extract_slice %main_params[%iv, 1, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
    %oweight = tensor.extract_slice %main_params[%iv, 0, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
    %obias   = tensor.extract_slice %main_params[%iv, 1, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
    %cweight = tensor.extract_slice %main_params[%iv, 0, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
    %cbias   = tensor.extract_slice %main_params[%iv, 1, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>

    %hidden = tensor.extract_slice %state[%iv, 0, 0] [1, 1, 14] [1, 1, 1] : tensor<2x2x14xf64> to tensor<14xf64>
    %cell   = tensor.extract_slice %state[%iv, 1, 0] [1, 1, 14] [1, 1, 1] : tensor<2x2x14xf64> to tensor<14xf64>

    // inlined lstm_model
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

    %state_0 = tensor.insert_slice %cell_new into %state[%iv, 1, 0] [1, 1, 14] [1, 1, 1] : tensor<14xf64> into tensor<2x2x14xf64>
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
    %state_next = tensor.insert_slice %hidden_next into %state_0[%iv, 0, 0] [1, 1, 14] [1, 1, 1] : tensor<14xf64> into tensor<2x2x14xf64>

    // end inlined lstm_model
    scf.yield %hidden_next, %state_next : tensor<14xf64>, tensor<2x2x14xf64>
  }

  %w2_1 = tensor.extract_slice %extra_params[1, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
  %w2_2 = tensor.extract_slice %extra_params[2, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
  %ypred = linalg.generic
    {
      indexing_maps = [#map, #map, #map, #map],
      iterator_types = ["parallel"]
    }
    ins(%xp#0, %w2_1, %w2_2 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>)
    outs(%zero_b : tensor<14xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    linalg.yield %1 : f64
  } -> tensor<14xf64>
  return %ypred : tensor<14xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %main_params = arith.constant dense<[
    [[[7.4776e-01, 2.6763e-01, 6.0353e-01, 8.9813e-01, 5.6639e-01, 7.0011e-01, 8.3863e-01, 2.7329e-01,
       2.1016e-01, 2.3980e-01, 7.5663e-01, 9.5941e-02, 8.3977e-01, 1.9742e-01],
      [3.7136e-01, 3.3427e-01, 7.1477e-01, 6.8785e-01, 3.8975e-01, 9.5685e-02, 7.9715e-01, 9.8289e-01,
       5.0200e-01, 2.0936e-01, 1.2142e-01, 5.2723e-01, 9.0605e-01, 1.0412e-01],
      [7.1775e-01, 7.3519e-02, 7.0361e-01, 5.8843e-01, 1.4773e-01, 3.4759e-01, 1.6100e-01, 2.9332e-01,
       2.5647e-02, 9.9352e-02, 4.9800e-01, 6.7219e-01, 6.7442e-03, 1.7286e-01],
      [7.2110e-01, 8.9900e-01, 4.8226e-01, 8.1050e-01, 9.7558e-01, 5.4716e-01, 8.0309e-01, 7.3151e-01,
       3.7093e-01, 2.0052e-01, 2.8181e-01, 5.1202e-01, 7.0023e-01, 7.1623e-01]],

     [[2.5894e-01, 9.7604e-02, 1.6987e-01, 5.0909e-01, 3.0241e-01, 8.7341e-01, 3.6577e-01, 3.3512e-01,
       6.5503e-01, 3.1045e-01, 6.8578e-01, 7.9193e-01, 4.9389e-01, 9.2962e-01],
      [8.2707e-01, 2.9147e-01, 3.5179e-01, 5.7766e-01, 5.7593e-01, 5.3437e-01, 9.3343e-01, 5.0185e-01,
       3.4979e-01, 6.6700e-01, 2.2466e-01, 3.5856e-01, 1.5230e-01, 8.0744e-01],
      [2.2909e-01, 7.9646e-01, 3.3835e-01, 6.9398e-01, 9.1655e-01, 2.0370e-02, 4.6953e-02, 5.3564e-01,
       5.1731e-01, 5.7201e-02, 6.0780e-01, 6.6295e-01, 6.0099e-01, 5.4842e-01],
      [2.1430e-01, 1.2279e-02, 4.1448e-02, 2.6355e-01, 8.1834e-01, 3.7592e-01, 8.0056e-01, 5.4941e-01,
       5.7665e-01, 1.4129e-01, 7.2936e-01, 9.3941e-01, 7.6730e-01, 7.9242e-01]]],


    [[[9.3046e-01, 6.1952e-01, 5.4884e-01, 6.7327e-01, 2.3048e-01, 2.7207e-01, 8.6120e-01, 5.9103e-01,
       5.7485e-01, 2.6422e-01, 6.5105e-01, 4.2774e-01, 8.3892e-01, 7.4174e-02],
      [9.2072e-01, 5.2067e-01, 9.8604e-01, 8.1995e-01, 3.1961e-01, 7.9856e-01, 7.4811e-01, 5.5811e-01,
       7.8895e-01, 9.5317e-01, 5.6595e-01, 7.1024e-01, 6.3305e-01, 2.2635e-03],
      [6.5955e-01, 5.9605e-01, 7.2530e-01, 1.6721e-01, 3.1531e-01, 6.0263e-01, 1.1245e-01, 3.3996e-01,
       3.3617e-01, 7.4906e-01, 8.1811e-01, 6.6861e-01, 1.4759e-01, 1.0455e-01],
      [1.7888e-01, 5.6079e-01, 5.8065e-01, 1.7768e-02, 1.9752e-01, 8.7258e-01, 1.8538e-02, 1.8181e-01,
       3.6392e-01, 9.2958e-01, 4.7798e-01, 6.1972e-01, 8.8105e-01, 4.1218e-01]],

     [[2.1684e-01, 9.9514e-02, 4.4734e-01, 2.1249e-01, 8.0572e-01, 3.5456e-01, 6.2826e-01, 8.4101e-01,
       4.0623e-01, 6.3289e-01, 9.8900e-01, 7.8349e-01, 5.1111e-01, 1.0322e-01],
      [9.4243e-01, 5.8501e-01, 2.7802e-01, 3.6256e-01, 9.2192e-02, 8.8976e-01, 4.7628e-02, 9.5051e-02,
       4.5935e-01, 2.0564e-01, 9.5959e-01, 9.4567e-01, 1.2757e-01, 8.1470e-01],
      [1.6030e-01, 5.5630e-01, 4.4646e-02, 9.5512e-02, 7.6818e-01, 2.4778e-01, 9.9085e-01, 4.0564e-01,
       3.4179e-01, 8.4640e-01, 4.1996e-01, 6.2428e-01, 4.3424e-01, 5.3619e-01],
      [6.5278e-01, 8.8175e-01, 4.6551e-01, 1.7662e-01, 8.6885e-01, 8.7139e-01, 1.1008e-01, 6.4424e-01,
       4.4923e-01, 4.1524e-02, 8.3894e-01, 8.0946e-01, 4.4153e-01, 9.5445e-01]]]]> : tensor<2x2x4x14xf64>
  %extra_params = arith.constant dense<[
    [5.8485e-01, 4.1080e-01, 1.9049e-01, 2.8525e-01, 1.9104e-01, 6.7381e-01, 1.8415e-03, 1.6199e-01, 1.1815e-01,
     2.4210e-01, 1.8944e-01, 1.2508e-01, 5.3325e-01, 2.9072e-01],
    [5.8577e-01, 4.0427e-01, 7.0894e-01, 7.2672e-01, 4.8060e-01, 6.3331e-01, 4.8448e-01, 4.7733e-01, 9.4118e-01,
     5.7953e-01, 4.6860e-01, 7.2396e-02, 1.4572e-01, 1.7834e-01],
    [7.4382e-01, 6.3325e-01, 7.4205e-02, 2.2704e-02, 8.0458e-01, 6.4519e-02, 5.6363e-01, 7.9363e-01, 2.6659e-01,
     8.2000e-01, 5.0870e-01, 6.1274e-01, 5.1378e-01, 1.1671e-01]]> : tensor<3x14xf64>
  %state_vals = arith.constant dense<[
    [[9.3682e-01, 4.8578e-01, 1.4065e-01, 1.6320e-01, 3.5769e-01, 2.9583e-02, 7.9633e-01, 8.4384e-01,
      3.3121e-01, 5.7096e-01, 6.9654e-01, 6.6078e-01, 9.5455e-01, 4.2876e-01],
     [2.5705e-01, 7.7617e-01, 8.6593e-01, 6.2958e-01, 6.5128e-01, 7.0054e-01, 1.8294e-01, 4.8103e-02,
      3.5343e-01, 1.7129e-01, 8.2196e-01, 2.2436e-01, 3.8642e-01, 6.7482e-01]],
    [[1.8866e-01, 3.6445e-01, 1.8349e-01, 9.4309e-01, 1.4140e-01, 1.4861e-01, 7.7937e-01, 2.2436e-02,
      9.0438e-01, 7.8972e-01, 9.2442e-01, 4.5663e-01, 2.2874e-02, 4.4151e-01],
     [5.6080e-01, 9.9277e-01, 2.0624e-01, 4.1726e-01, 4.4307e-01, 7.3984e-01, 1.8457e-01, 1.8016e-01,
      1.8162e-01, 4.5309e-01, 3.0257e-01, 5.7440e-02, 4.3623e-01, 2.6003e-01]]]> : tensor<2x2x14xf64>
  // %state_vals = arith.constant dense<[
  //   9.3682e-01, 4.8578e-01, 1.4065e-01, 1.6320e-01, 3.5769e-01, 2.9583e-02, 7.9633e-01, 8.4384e-01,
  //     3.3121e-01, 5.7096e-01, 6.9654e-01, 6.6078e-01, 9.5455e-01, 4.2876e-01,
  //   2.5705e-01, 7.7617e-01, 8.6593e-01, 6.2958e-01, 6.5128e-01, 7.0054e-01, 1.8294e-01, 4.8103e-02,
  //     3.5343e-01, 1.7129e-01, 8.2196e-01, 2.2436e-01, 3.8642e-01, 6.7482e-01,
  //   1.8866e-01, 3.6445e-01, 1.8349e-01, 9.4309e-01, 1.4140e-01, 1.4861e-01, 7.7937e-01, 2.2436e-02,
  //     9.0438e-01, 7.8972e-01, 9.2442e-01, 4.5663e-01, 2.2874e-02, 4.4151e-01,
  //   5.6080e-01, 9.9277e-01, 2.0624e-01, 4.1726e-01, 4.4307e-01, 7.3984e-01, 1.8457e-01, 1.8016e-01,
  //     1.8162e-01, 4.5309e-01, 3.0257e-01, 5.7440e-02, 4.3623e-01, 2.6003e-01]> : tensor<2x2x14xf64>
  %x = arith.constant dense<[0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.]> : tensor<14xf64>
  %zero_state = arith.constant dense<0.0> : tensor<2x2x14xf64>
  %state_init = arith.addf %state_vals, %zero_state : tensor<2x2x14xf64>
  // %primal = call @lstm_predict(%main_params, %extra_params, %state_vals, %x) : (tensor<2x2x4x14xf64>, tensor<3x14xf64>, tensor<2x2x14xf64>, tensor<14xf64>) -> tensor<14xf64>
  // %U = tensor.cast %primal : tensor<14xf64> to tensor<*xf64>

  %f = constant @lstm_predict : (tensor<2x2x4x14xf64>, tensor<3x14xf64>, tensor<2x2x14xf64>, tensor<14xf64>) -> tensor<14xf64>
  %df = standalone.grad %f {of = [0, 1, 2]} : (
    tensor<2x2x4x14xf64>,
    tensor<3x14xf64>,
    tensor<2x2x14xf64>,
    tensor<14xf64>
  ) -> tensor<14xf64>, (
    tensor<2x2x4x14xf64>,
    tensor<3x14xf64>,
    tensor<2x2x14xf64>,
    tensor<14xf64>
  ) -> (tensor<2x2x4x14xf64>, tensor<3x14xf64>, tensor<2x2x14xf64>)
  %res:3 = call_indirect %df(%main_params, %extra_params, %state_init, %x) : (
    tensor<2x2x4x14xf64>,
    tensor<3x14xf64>,
    tensor<2x2x14xf64>,
    tensor<14xf64>
  ) -> (tensor<2x2x4x14xf64>, tensor<3x14xf64>, tensor<2x2x14xf64>)
  // %U = tensor.cast %res#0 : tensor<2x2x4x14xf64> to tensor<*xf64>
  // %U = tensor.cast %res#1 : tensor<3x14xf64> to tensor<*xf64>
  %Us = tensor.cast %res#2 : tensor<2x2x14xf64> to tensor<*xf64>
  call @print_memref_f64(%Us) : (tensor<*xf64>) -> ()
  return
}
