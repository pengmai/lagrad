#map = affine_map<(d0) -> (d0)>

func private @msigmoid(%x: f64) -> f64 {
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

func @lagrad_lstm_predict(%main_params: tensor<2x2x4x14xf64>, %extra_params: tensor<3x14xf64>, %state_init: tensor<2x2x14xf64>, %x: tensor<14xf64>) -> (
  tensor<2x2x4x14xf64>,
  tensor<3x14xf64>,
  tensor<2x2x14xf64>
) {
  %res:3 = lagrad.grad @lstm_predict(%main_params, %extra_params, %state_init, %x) {of = [0, 1, 2]} : (
    tensor<2x2x4x14xf64>,
    tensor<3x14xf64>,
    tensor<2x2x14xf64>,
    tensor<14xf64>
  ) -> (tensor<2x2x4x14xf64>, tensor<3x14xf64>, tensor<2x2x14xf64>)
  return %res#0, %res#1, %res#2 : tensor<2x2x4x14xf64>, tensor<3x14xf64>, tensor<2x2x14xf64>
}
