#map = affine_map<(d0) -> (d0)>

func private @msigmoid(%x: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %nx = arith.negf %x : f64
  %exp = math.exp %nx : f64
  %denom = arith.addf %one, %exp : f64
  %frac = arith.divf %one, %denom : f64
  return %frac : f64
}

func private @tsigmoid(%x: tensor<{{b}}xf64>) -> tensor<{{b}}xf64> {
  %one = arith.constant 1.0 : f64
  %zerot = arith.constant dense<0.0> : tensor<{{b}}xf64>
  %ts = linalg.generic
    { indexing_maps = [#map, #map], iterator_types = ["parallel"] }
    ins(%x : tensor<{{b}}xf64>)
    outs(%zerot : tensor<{{b}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %nx = arith.negf %arg0 : f64
    %exp = math.exp %nx : f64
    %denom = arith.addf %one, %exp : f64
    %frac = arith.divf %one, %denom : f64
    linalg.yield %frac : f64
  } -> tensor<{{b}}xf64>
  return %ts : tensor<{{b}}xf64>
}

func @mlogsumexp(%t: tensor<{{b}}xf64>) -> f64 {
  %out_init = arith.constant dense<0.0> : tensor<f64>
  %lse = linalg.generic
    {
      indexing_maps = [#map, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    }
    ins(%t : tensor<{{b}}xf64>)
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

// func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @mlstm_objective(
  %main_params: tensor<{{l}}x2x4x{{b}}xf64>,
  %extra_params: tensor<3x{{b}}xf64>,
  %state_init: tensor<{{l}}x2x{{b}}xf64>,
  %sequence: tensor<{{c}}x{{b}}xf64>
) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %cb = arith.constant {{b}} : index
  %cc = arith.constant {{c}} : index
  %ccm1 = arith.constant {{c - 1}} : index
  %cl = arith.constant {{l}} : index
  %zero = arith.constant 0.0 : f64
  %zerod_t = arith.constant dense<0.0> : tensor<f64>
  %zero_b = arith.constant dense<0.0> : tensor<{{b}}xf64>
  %res:3 = scf.for %t = %c0 to %ccm1 step %c1 iter_args(%total = %zero, %count = %c0, %state_outer = %state_init) -> (f64, index, tensor<{{l}}x2x{{b}}xf64>) {
    // inlined lstm predict
    %x = tensor.extract_slice %sequence[%t, 0] [1, {{b}}] [1, 1] : tensor<{{c}}x{{b}}xf64> to tensor<{{b}}xf64>
    %w2 = tensor.extract_slice %extra_params[0, 0] [1, {{b}}] [1, 1] : tensor<3x{{b}}xf64> to tensor<{{b}}xf64>
    %x2 = arith.mulf %x, %w2 : tensor<{{b}}xf64>

    %xp:2 = scf.for %iv = %c0 to %cl step %c1 iter_args(%input = %x2, %state = %state_outer) -> (tensor<{{b}}xf64>, tensor<{{l}}x2x{{b}}xf64>) {
      %fweight = tensor.extract_slice %main_params[%iv, 0, 0, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : tensor<{{l}}x2x4x{{b}}xf64> to tensor<{{b}}xf64>
      %fbias   = tensor.extract_slice %main_params[%iv, 1, 0, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : tensor<{{l}}x2x4x{{b}}xf64> to tensor<{{b}}xf64>
      %iweight = tensor.extract_slice %main_params[%iv, 0, 1, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : tensor<{{l}}x2x4x{{b}}xf64> to tensor<{{b}}xf64>
      %ibias   = tensor.extract_slice %main_params[%iv, 1, 1, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : tensor<{{l}}x2x4x{{b}}xf64> to tensor<{{b}}xf64>
      %oweight = tensor.extract_slice %main_params[%iv, 0, 2, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : tensor<{{l}}x2x4x{{b}}xf64> to tensor<{{b}}xf64>
      %obias   = tensor.extract_slice %main_params[%iv, 1, 2, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : tensor<{{l}}x2x4x{{b}}xf64> to tensor<{{b}}xf64>
      %cweight = tensor.extract_slice %main_params[%iv, 0, 3, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : tensor<{{l}}x2x4x{{b}}xf64> to tensor<{{b}}xf64>
      %cbias   = tensor.extract_slice %main_params[%iv, 1, 3, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : tensor<{{l}}x2x4x{{b}}xf64> to tensor<{{b}}xf64>

      %hidden = tensor.extract_slice %state[%iv, 0, 0] [1, 1, {{b}}] [1, 1, 1] : tensor<{{l}}x2x{{b}}xf64> to tensor<{{b}}xf64>
      %cell   = tensor.extract_slice %state[%iv, 1, 0] [1, 1, {{b}}] [1, 1, 1] : tensor<{{l}}x2x{{b}}xf64> to tensor<{{b}}xf64>

      // inlined lstm_model
      // %forget = linalg.generic
      //   {
      //     indexing_maps = [#map, #map, #map, #map],
      //     iterator_types = ["parallel"]
      //   }
      //   ins(%input, %fweight, %fbias : tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>)
      //   outs(%zero_b : tensor<{{b}}xf64>) {
      // ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
      //   %0 = arith.mulf %arg0, %arg1 : f64
      //   %1 = arith.addf %0, %arg2 : f64
      //   %2 = call @msigmoid(%1) : (f64) -> f64
      //   linalg.yield %2 : f64
      // } -> tensor<{{b}}xf64>
      %forget0 = arith.mulf %input, %fweight : tensor<{{b}}xf64>
      %forget1 = arith.addf %forget0, %fbias : tensor<{{b}}xf64>
      %forget = call @tsigmoid(%forget1) : (tensor<{{b}}xf64>) -> tensor<{{b}}xf64>

      // %ingate = linalg.generic
      //   {
      //     indexing_maps = [#map, #map, #map, #map],
      //     iterator_types = ["parallel"]
      //   }
      //   ins(%hidden, %iweight, %ibias : tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>)
      //   outs(%zero_b : tensor<{{b}}xf64>) {
      // ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
      //   %0 = arith.mulf %arg0, %arg1 : f64
      //   %1 = arith.addf %0, %arg2 : f64
      //   %2 = call @msigmoid(%1) : (f64) -> f64
      //   linalg.yield %2 : f64
      // } -> tensor<{{b}}xf64>
      %ingate0 = arith.mulf %hidden, %iweight : tensor<{{b}}xf64>
      %ingate1 = arith.addf %ingate0, %ibias : tensor<{{b}}xf64>
      %ingate = call @tsigmoid(%ingate1) : (tensor<{{b}}xf64>) -> tensor<{{b}}xf64>

      // %outgate = linalg.generic
      //   {
      //     indexing_maps = [#map, #map, #map, #map],
      //     iterator_types = ["parallel"]
      //   }
      //   ins(%input, %oweight, %obias : tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>)
      //   outs(%zero_b : tensor<{{b}}xf64>) {
      // ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
      //   %0 = arith.mulf %arg0, %arg1 : f64
      //   %1 = arith.addf %0, %arg2 : f64
      //   %2 = call @msigmoid(%1) : (f64) -> f64
      //   linalg.yield %2 : f64
      // } -> tensor<{{b}}xf64>
      %outgate0 = arith.mulf %input, %oweight : tensor<{{b}}xf64>
      %outgate1 = arith.addf %outgate0, %obias : tensor<{{b}}xf64>
      %outgate = call @tsigmoid(%outgate1) : (tensor<{{b}}xf64>) -> tensor<{{b}}xf64>
      // %change = linalg.generic
      //   {
      //     indexing_maps = [#map, #map, #map, #map],
      //     iterator_types = ["parallel"]
      //   }
      //   ins(%hidden, %cweight, %cbias : tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>)
      //   outs(%zero_b : tensor<{{b}}xf64>) {
      // ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
      //   %0 = arith.mulf %arg0, %arg1 : f64
      //   %1 = arith.addf %0, %arg2 : f64
      //   %2 = math.tanh %1 : f64
      //   linalg.yield %2 : f64
      // } -> tensor<{{b}}xf64>
      %change0 = arith.mulf %hidden, %cweight : tensor<{{b}}xf64>
      %change1 = arith.addf %change0, %cbias : tensor<{{b}}xf64>
      %change = math.tanh %change1 : tensor<{{b}}xf64>

      // %cell_new = linalg.generic
      //   {
      //     indexing_maps = [#map, #map, #map, #map, #map],
      //     iterator_types = ["parallel"]
      //   }
      //   ins(
      //     %cell,
      //     %forget,
      //     %ingate,
      //     %change : tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>
      //   )
      //   outs(%zero_b : tensor<{{b}}xf64>) {
      // ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64):
      //   %0 = arith.mulf %arg0, %arg1 : f64
      //   %1 = arith.mulf %arg2, %arg3 : f64
      //   %2 = arith.addf %0, %1 : f64
      //   linalg.yield %2 : f64
      // } -> tensor<{{b}}xf64>
      %cell_new0 = arith.mulf %cell, %forget : tensor<{{b}}xf64>
      %cell_new1 = arith.mulf %ingate, %change : tensor<{{b}}xf64>
      %cell_new = arith.addf %cell_new0, %cell_new1 : tensor<{{b}}xf64>

      %state_0 = tensor.insert_slice %cell_new into %state[%iv, 1, 0] [1, 1, {{b}}] [1, 1, 1] : tensor<{{b}}xf64> into tensor<{{l}}x2x{{b}}xf64>
      // %hidden_next = linalg.generic
      //   {
      //     indexing_maps = [#map, #map, #map],
      //     iterator_types = ["parallel"]
      //   }
      //   ins(%outgate, %cell_new : tensor<{{b}}xf64>, tensor<{{b}}xf64>)
      //   outs(%zero_b : tensor<{{b}}xf64>) {
      // ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      //   %0 = math.tanh %arg1 : f64
      //   %1 = arith.mulf %arg0, %0 : f64
      //   linalg.yield %1 : f64
      // } -> tensor<{{b}}xf64>
      %hidden_next0 = math.tanh %cell_new : tensor<{{b}}xf64>
      %hidden_next = arith.mulf %outgate, %hidden_next0 : tensor<{{b}}xf64>
      %state_next = tensor.insert_slice %hidden_next into %state_0[%iv, 0, 0] [1, 1, {{b}}] [1, 1, 1] : tensor<{{b}}xf64> into tensor<{{l}}x2x{{b}}xf64>

      // end inlined lstm_model
      scf.yield %hidden_next, %state_next : tensor<{{b}}xf64>, tensor<{{l}}x2x{{b}}xf64>
    }

    %w2_1 = tensor.extract_slice %extra_params[1, 0] [1, {{b}}] [1, 1] : tensor<3x{{b}}xf64> to tensor<{{b}}xf64>
    %w2_2 = tensor.extract_slice %extra_params[2, 0] [1, {{b}}] [1, 1] : tensor<3x{{b}}xf64> to tensor<{{b}}xf64>
    // %ypred = linalg.generic
    //   {
    //     indexing_maps = [#map, #map, #map, #map],
    //     iterator_types = ["parallel"]
    //   }
    //   ins(%xp#0, %w2_1, %w2_2 : tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>)
    //   outs(%zero_b : tensor<{{b}}xf64>) {
    // ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    //   %0 = arith.mulf %arg0, %arg1 : f64
    //   %1 = arith.addf %0, %arg2 : f64
    //   linalg.yield %1 : f64
    // } -> tensor<{{b}}xf64>
    %ypred0 = arith.mulf %xp#0, %w2_1 : tensor<{{b}}xf64>
    %ypred = arith.addf %ypred0, %w2_2 : tensor<{{b}}xf64>

    // end inlined lstm predict
    %lse = call @mlogsumexp(%ypred) : (tensor<{{b}}xf64>) -> f64
    %ynorm = linalg.generic
      {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      }
      ins(%ypred : tensor<{{b}}xf64>)
      outs(%zero_b : tensor<{{b}}xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.subf %arg0, %lse : f64
      linalg.yield %0 : f64
    } -> tensor<{{b}}xf64>
    %ygold_idx = arith.addi %t, %c1 : index
    %ygold = tensor.extract_slice %sequence[%ygold_idx, 0] [1, {{b}}] [1, 1] : tensor<{{c}}x{{b}}xf64> to tensor<{{b}}xf64>

    %total_i = linalg.dot ins(%ygold, %ynorm : tensor<{{b}}xf64>, tensor<{{b}}xf64>) outs(%zerod_t : tensor<f64>) -> tensor<f64>
    %total_i_val = tensor.extract %total_i[] : tensor<f64>
    %total_next = arith.addf %total_i_val, %total : f64
    %count_next = arith.addi %count, %cb : index
    scf.yield %total_next, %count_next, %xp#1 : f64, index, tensor<{{l}}x2x{{b}}xf64>
  }
  %ntotal = arith.negf %res#0 : f64
  %counti = arith.index_cast %res#1 : index to i64
  %countf = arith.sitofp %counti : i64 to f64
  %loss = arith.divf %ntotal, %countf : f64
  return %loss : f64
}

func @lagrad_lstm(
  %main_params: tensor<{{l}}x2x4x{{b}}xf64>,
  %extra_params: tensor<3x{{b}}xf64>,
  %state_init: tensor<{{l}}x2x{{b}}xf64>,
  %sequence: tensor<{{c}}x{{b}}xf64>
) -> (tensor<{{l}}x2x4x{{b}}xf64>, tensor<3x{{b}}xf64>) {
  // return %main_params, %extra_params : tensor<{{l}}x2x4x{{b}}xf64>, tensor<3x{{b}}xf64>
  %f = constant @mlstm_objective : (
    tensor<{{l}}x2x4x{{b}}xf64>,
    tensor<3x{{b}}xf64>,
    tensor<{{l}}x2x{{b}}xf64>,
    tensor<{{c}}x{{b}}xf64>
  ) -> f64
  %df = standalone.grad %f {of = [0, 1]} : (
    tensor<{{l}}x2x4x{{b}}xf64>,
    tensor<3x{{b}}xf64>,
    tensor<{{l}}x2x{{b}}xf64>,
    tensor<{{c}}x{{b}}xf64>
  ) -> f64, (
    tensor<{{l}}x2x4x{{b}}xf64>,
    tensor<3x{{b}}xf64>,
    tensor<{{l}}x2x{{b}}xf64>,
    tensor<{{c}}x{{b}}xf64>
  ) -> (tensor<{{l}}x2x4x{{b}}xf64>, tensor<3x{{b}}xf64>)

  %res:2 = call_indirect %df(%main_params, %extra_params, %state_init, %sequence) : (
    tensor<{{l}}x2x4x{{b}}xf64>,
    tensor<3x{{b}}xf64>,
    tensor<{{l}}x2x{{b}}xf64>,
    tensor<{{c}}x{{b}}xf64>
  ) -> (tensor<{{l}}x2x4x{{b}}xf64>, tensor<3x{{b}}xf64>)
  return %res#0, %res#1 : tensor<{{l}}x2x4x{{b}}xf64>, tensor<3x{{b}}xf64>
}
