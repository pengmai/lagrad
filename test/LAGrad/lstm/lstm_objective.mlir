#map = affine_map<(d0) -> (d0)>

func.func private @msigmoid(%x: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %nx = arith.negf %x : f64
  %exp = math.exp %nx : f64
  %denom = arith.addf %one, %exp : f64
  %frac = arith.divf %one, %denom : f64
  return %frac : f64
}

func.func @mlogsumexp(%t: tensor<14xf64>) -> f64 {
  %out_init = arith.constant dense<0.0> : tensor<f64>
  %lse = linalg.generic
    {
      indexing_maps = [#map, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    }
    ins(%t : tensor<14xf64>)
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

func.func @mlstm_objective(
  %main_params: tensor<2x2x4x14xf64>,
  %extra_params: tensor<3x14xf64>,
  %state_init: tensor<2x2x14xf64>,
  %sequence: tensor<4x14xf64>
) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %cb = arith.constant 14 : index
  %cc = tensor.dim %sequence, %c0 : tensor<4x14xf64>
  %ccm1 = arith.subi %cc, %c1 : index
  %cl = arith.constant 2 : index
  %zero = arith.constant 0.0 : f64
  %zerod_t = arith.constant dense<0.0> : tensor<f64>
  %zero_b = arith.constant dense<0.0> : tensor<14xf64>
  %res:3 = scf.for %t = %c0 to %ccm1 step %c1 iter_args(%total = %zero, %count = %c0, %state_outer = %state_init) -> (f64, index, tensor<2x2x14xf64>) {
    // inlined lstm predict
    %x = tensor.extract_slice %sequence[%t, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
    %w2 = tensor.extract_slice %extra_params[0, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
    %x2 = arith.mulf %x, %w2 : tensor<14xf64>

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
        %2 = func.call @msigmoid(%1) : (f64) -> f64
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
        %2 = func.call @msigmoid(%1) : (f64) -> f64
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
        %2 = func.call @msigmoid(%1) : (f64) -> f64
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

      // how do I know that this is supposed to be in-place?
      // A slice of dstate needs to be added to dcell_new.
      // env[%cell_new] += dstate[%iv, 1]
      %state_0 = tensor.insert_slice %cell_new into %state[%iv, 1, 0] [1, 1, 14] [1, 1, 1] {debugme = true} : tensor<14xf64> into tensor<2x2x14xf64>

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
      // there's a bug here. %state_next doesn't have any incoming gradient signal through its dest() nor its result().
      // However, %state_0 does have a gradient signal from %state.
      %state_next = tensor.insert_slice %hidden_next into %state_0[%iv, 0, 0] [1, 1, 14] [1, 1, 1] : tensor<14xf64> into tensor<2x2x14xf64>
      // uncommenting these two lines will probably break autodiff. Or at least give the incorrect result.
      // Actually this is causing an issue in the grad, will need to fix it.
      // %state_0 = tensor.insert_slice %hidden_next into %state[%iv, 0, 0] [1, 1, 14] [1, 1, 1] : tensor<14xf64> into tensor<2x2x14xf64>
      // %state_next = tensor.insert_slice %cell_new into %state_0[%iv, 1, 0] [1, 1, 14] [1, 1, 1] {debugme = true} : tensor<14xf64> into tensor<2x2x14xf64>
      %input_next = arith.addf %hidden_next, %zero_b : tensor<14xf64>

      // end inlined lstm_model
      scf.yield %input_next, %state_next : tensor<14xf64>, tensor<2x2x14xf64>
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

    // end inlined lstm predict
    %lse = func.call @mlogsumexp(%ypred) : (tensor<14xf64>) -> f64
    %ynorm = linalg.generic
      {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      }
      ins(%ypred : tensor<14xf64>)
      outs(%zero_b : tensor<14xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.subf %arg0, %lse : f64
      linalg.yield %0 : f64
    } -> tensor<14xf64>
    %ygold_idx = arith.addi %t, %c1 : index
    %ygold = tensor.extract_slice %sequence[%ygold_idx, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>

    %total_i = linalg.dot ins(%ygold, %ynorm : tensor<14xf64>, tensor<14xf64>) outs(%zerod_t : tensor<f64>) -> tensor<f64>
    %total_i_val = tensor.extract %total_i[] : tensor<f64>
    %total_next = arith.addf %total_i_val, %total : f64
    %count_next = arith.addi %count, %cb : index
    scf.yield %total_next, %count_next, %xp#1 : f64, index, tensor<2x2x14xf64>
  }
  %ntotal = arith.negf %res#0 : f64
  %counti = arith.index_cast %res#1 : index to i64
  %countf = arith.sitofp %counti : i64 to f64
  %loss = arith.divf %ntotal, %countf : f64
  return %loss : f64
}

// func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func.func @lagrad_lstm_objective(
  %main_params: tensor<2x2x4x14xf64>,
  %extra_params: tensor<3x14xf64>,
  %state_init: tensor<2x2x14xf64>,
  %sequence: tensor<4x14xf64>
) -> (tensor<2x2x4x14xf64>, tensor<3x14xf64>) {
  %res:2 = lagrad.grad @mlstm_objective(%main_params, %extra_params, %state_init, %sequence) {of = [0, 1]} : (
    tensor<2x2x4x14xf64>,
    tensor<3x14xf64>,
    tensor<2x2x14xf64>,
    tensor<4x14xf64>
  ) -> (tensor<2x2x4x14xf64>, tensor<3x14xf64>)
  return %res#0, %res#1 : tensor<2x2x4x14xf64>, tensor<3x14xf64>
}
