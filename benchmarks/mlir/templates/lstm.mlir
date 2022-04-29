#map = affine_map<(d0) -> (d0)>

func @msigmoid(%x: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %nx = arith.negf %x : f64
  %exp = math.exp %nx : f64
  %denom = arith.addf %one, %exp : f64
  %frac = arith.divf %one, %denom : f64
  return %frac : f64
}

func @mlogsumexp(%t: tensor<{{b}}xf64>) -> f64 {
  %out_init = arith.constant dense<0.0> : tensor<f64>
  %lse = linalg.generic
    {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
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

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @mlstm_objective(
  %main_params: tensor<{{main_sz}}xf64>,
  %extra_params: tensor<{{extra_sz}}xf64>,
  %state_init: tensor<{{state_sz}}xf64>,
  %sequence: tensor<{{seq_sz}}xf64>
) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %cb = arith.constant {{b}} : index
  %cc = arith.constant {{c}} : index
  %cl = arith.constant {{l}} : index
  %twob = arith.muli %cb, %c2 : index
  %twob_l = arith.muli %twob, %cl : index
  %cc_minus_one = arith.subi %cc, %c1 : index
  %ub = arith.muli %cc_minus_one, %cb : index // The original C implementation uses a leq here when checking loop bounds
  %zero = arith.constant 0.0 : f64
  %zerod_t = arith.constant dense<0.0> : tensor<f64>
  %zero_b = arith.constant dense<0.0> : tensor<{{b}}xf64>
  // %forget_space = linalg.init_tensor [{{b}}] : tensor<{{b}}xf64>
  // %ingate_space = linalg.init_tensor [{{b}}] : tensor<{{b}}xf64>
  // %outgate_space = linalg.init_tensor [{{b}}] : tensor<{{b}}xf64>
  // %change_space = linalg.init_tensor [{{b}}] : tensor<{{b}}xf64>
  // %ynorm_space = linalg.init_tensor [{{b}}] : tensor<{{b}}xf64>
  %res:3 = scf.for %t = %c0 to %ub step %cb iter_args(%total = %zero, %count = %c0, %state_outer = %state_init) -> (f64, index, tensor<{{state_sz}}xf64>) {
    // inlined lstm predict
    %x = tensor.extract_slice %sequence[%t] [{{b}}] [1] : tensor<{{seq_sz}}xf64> to tensor<{{b}}xf64>
    %w2 = tensor.extract_slice %extra_params[0] [{{b}}] [1] : tensor<{{extra_sz}}xf64> to tensor<{{b}}xf64>
    %x2 = arith.mulf %x, %w2 : tensor<{{b}}xf64>

    // %const = arith.constant {{b*10}} : index
    // %pred = arith.cmpi eq, %const, %t : index
    // scf.if %pred {
    //   %U = tensor.cast %x2 : tensor<{{b}}xf64> to tensor<*xf64>
    //   call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
    // }
    %xp:2 = scf.for %iv = %c0 to %twob_l step %twob iter_args(%input = %x2, %state = %state_outer) -> (tensor<{{b}}xf64>, tensor<{{state_sz}}xf64>) {
      // In the second iteration, %input is incorrect.
      %first_outer_iter = arith.cmpi eq, %t, %cb : index

      %hidden = tensor.extract_slice %state[%iv] [{{b}}] [1] : tensor<{{state_sz}}xf64> to tensor<{{b}}xf64>

      scf.if %first_outer_iter {
        %U = tensor.cast %hidden : tensor<{{b}}xf64> to tensor<*xf64>
        call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
      }
      %w_idx = arith.muli %iv, %c4 : index
      %weights = tensor.extract_slice %main_params[%w_idx] [{{b}}] [1] : tensor<{{main_sz}}xf64> to tensor<{{b}}xf64>
      %iv_b = arith.addi %iv, %cb : index
      %b_idx = arith.muli %iv_b, %c4 : index
      %biases = tensor.extract_slice %main_params[%b_idx] [{{b}}] [1] : tensor<{{main_sz}}xf64> to tensor<{{b}}xf64>
      // inlined lstm_model
      %forget = linalg.generic
        {
          indexing_maps = [#map, #map, #map, #map],
          iterator_types = ["parallel"]
        }
        ins(%input, %weights, %biases : tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>)
        outs(%zero_b : tensor<{{b}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        %1 = arith.addf %0, %arg2 : f64
        %2 = call @msigmoid(%1) : (f64) -> f64
        linalg.yield %2 : f64
      } -> tensor<{{b}}xf64>
      %w_idx_next = arith.addi %w_idx, %cb : index
      %b_idx_next = arith.addi %b_idx, %cb : index
      // %hidden = tensor.extract_slice %state[%iv] [{{b}}] [1] : tensor<{{state_sz}}xf64> to tensor<{{b}}xf64>
      %weights_next = tensor.extract_slice %main_params[%w_idx_next] [{{b}}] [1] : tensor<{{main_sz}}xf64> to tensor<{{b}}xf64>
      %biases_next = tensor.extract_slice %main_params[%b_idx_next] [{{b}}] [1] : tensor<{{main_sz}}xf64> to tensor<{{b}}xf64>
      %ingate = linalg.generic
        {
          indexing_maps = [#map, #map, #map, #map],
          iterator_types = ["parallel"]
        }
        ins(%hidden, %weights_next, %biases_next : tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>)
        outs(%zero_b : tensor<{{b}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        %1 = arith.addf %0, %arg2 : f64
        %2 = call @msigmoid(%1) : (f64) -> f64
        linalg.yield %2 : f64
      } -> tensor<{{b}}xf64>
      %w_idx_2 = arith.addi %w_idx_next, %cb : index
      %b_idx_2 = arith.addi %b_idx_next, %cb : index
      %outweights = tensor.extract_slice %main_params[%w_idx_2] [{{b}}] [1] : tensor<{{main_sz}}xf64> to tensor<{{b}}xf64>
      %outbiases = tensor.extract_slice %main_params[%b_idx_2] [{{b}}] [1] : tensor<{{main_sz}}xf64> to tensor<{{b}}xf64>
      %outgate = linalg.generic
       {
          indexing_maps = [#map, #map, #map, #map],
          iterator_types = ["parallel"]
        }
        ins(%input, %outweights, %outbiases : tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>)
        outs(%zero_b : tensor<{{b}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        %1 = arith.addf %0, %arg2 : f64
        %2 = call @msigmoid(%1) : (f64) -> f64
        linalg.yield %2 : f64
      } -> tensor<{{b}}xf64>
      %w_idx_3 = arith.addi %w_idx_2, %cb : index
      %b_idx_3 = arith.addi %b_idx_2, %cb : index
      %changeweights = tensor.extract_slice %main_params[%w_idx_3] [{{b}}] [1] : tensor<{{main_sz}}xf64> to tensor<{{b}}xf64>
      %changebiases = tensor.extract_slice %main_params[%b_idx_3] [{{b}}] [1] : tensor<{{main_sz}}xf64> to tensor<{{b}}xf64>
      %change = linalg.generic
        {
          indexing_maps = [#map, #map, #map, #map],
          iterator_types = ["parallel"]
        }
        ins(%hidden, %changeweights, %changebiases : tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>)
        outs(%zero_b : tensor<{{b}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        %1 = arith.addf %0, %arg2 : f64
        %2 = math.tanh %1 : f64
        linalg.yield %2 : f64
      } -> tensor<{{b}}xf64>

      %cell_old = tensor.extract_slice %state[%iv_b] [{{b}}] [1] : tensor<{{state_sz}}xf64> to tensor<{{b}}xf64>
      %cell = linalg.generic
        {
          indexing_maps = [#map, #map, #map, #map, #map],
          iterator_types = ["parallel"]
        }
        ins(
          %cell_old,
          %forget,
          %ingate,
          %change : tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>
        )
        outs(%zero_b : tensor<{{b}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        %1 = arith.mulf %arg2, %arg3 : f64
        %2 = arith.addf %0, %1 : f64
        linalg.yield %2 : f64
      } -> tensor<{{b}}xf64>

      %state_0 = tensor.insert_slice %cell into %state[%iv_b] [{{b}}] [1] : tensor<{{b}}xf64> into tensor<{{state_sz}}xf64>
      %hidden_next = linalg.generic
        {
          indexing_maps = [#map, #map, #map],
          iterator_types = ["parallel"]
        }
        ins(%outgate, %cell : tensor<{{b}}xf64>, tensor<{{b}}xf64>)
        outs(%zero_b : tensor<{{b}}xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = math.tanh %arg1 : f64
        %1 = arith.mulf %arg0, %0 : f64
        linalg.yield %1 : f64
      } -> tensor<{{b}}xf64>
      %state_next = tensor.insert_slice %hidden_next into %state_0[%iv] [{{b}}] [1] : tensor<{{b}}xf64> into tensor<{{state_sz}}xf64>

      // end inlined lstm_model
      scf.yield %hidden_next, %state_next : tensor<{{b}}xf64>, tensor<{{state_sz}}xf64>
    }

    %w2_1 = tensor.extract_slice %extra_params[%cb] [{{b}}] [1] : tensor<{{extra_sz}}xf64> to tensor<{{b}}xf64>
    %w2_2 = tensor.extract_slice %extra_params[%twob] [{{b}}] [1] : tensor<{{extra_sz}}xf64> to tensor<{{b}}xf64>
    %ypred = linalg.generic
      {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      }
      ins(%xp#0, %w2_1, %w2_2 : tensor<{{b}}xf64>, tensor<{{b}}xf64>, tensor<{{b}}xf64>)
      outs(%zero_b : tensor<{{b}}xf64>) {
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
      %0 = arith.mulf %arg0, %arg1 : f64
      %1 = arith.addf %0, %arg2 : f64
      linalg.yield %1 : f64
    } -> tensor<{{b}}xf64>

    // end inlined lstm predict
    // %ypred = arith.constant dense<1.0> : tensor<{{b}}xf64>
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
    %ygold_idx = arith.addi %t, %cb : index
    %ygold = tensor.extract_slice %sequence[%ygold_idx] [{{b}}] [1] : tensor<{{seq_sz}}xf64> to tensor<{{b}}xf64>

    %total_i = linalg.dot ins(%ygold, %ynorm : tensor<{{b}}xf64>, tensor<{{b}}xf64>) outs(%zerod_t : tensor<f64>) -> tensor<f64>
    %total_i_val = tensor.extract %total_i[] : tensor<f64>
    %total_next = arith.addf %total_i_val, %total : f64
    %count_next = arith.addi %count, %cb : index
    scf.yield %total_next, %count_next, %xp#1 : f64, index, tensor<{{state_sz}}xf64>
  }
  %ntotal = arith.negf %res#0 : f64
  %counti = arith.index_cast %res#1 : index to i64
  %countf = arith.sitofp %counti : i64 to f64
  %loss = arith.divf %ntotal, %countf : f64
  return %loss : f64
}

func @lagrad_lstm(
  %main_params: tensor<{{main_sz}}xf64>,
  %extra_params: tensor<{{extra_sz}}xf64>,
  %state: tensor<{{state_sz}}xf64>,
  %sequence: tensor<{{seq_sz}}xf64>
) -> (tensor<{{main_sz}}xf64>, tensor<{{extra_sz}}xf64>) {
  %f = constant @mlstm_objective : (
    tensor<{{main_sz}}xf64>,
    tensor<{{extra_sz}}xf64>,
    tensor<{{state_sz}}xf64>,
    tensor<{{seq_sz}}xf64>
  ) -> f64
  %df = standalone.grad %f {of = [0, 1]} : (
    tensor<{{main_sz}}xf64>,
    tensor<{{extra_sz}}xf64>,
    tensor<{{state_sz}}xf64>,
    tensor<{{seq_sz}}xf64>
  ) -> f64, (
    tensor<{{main_sz}}xf64>,
    tensor<{{extra_sz}}xf64>,
    tensor<{{state_sz}}xf64>,
    tensor<{{seq_sz}}xf64>
  ) -> (tensor<{{main_sz}}xf64>, tensor<{{extra_sz}}xf64>)

  %res:2 = call_indirect %df(%main_params, %extra_params, %state, %sequence) : (
    tensor<{{main_sz}}xf64>,
    tensor<{{extra_sz}}xf64>,
    tensor<{{state_sz}}xf64>,
    tensor<{{seq_sz}}xf64>
  ) -> (tensor<{{main_sz}}xf64>, tensor<{{extra_sz}}xf64>)
  return %res#0, %res#1 : tensor<{{main_sz}}xf64>, tensor<{{extra_sz}}xf64>
}
