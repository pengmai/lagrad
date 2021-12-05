// A microbenchmark from the hot part of GMMs
#id_1d = affine_map<(d0) -> (d0)>
#reduce_1d = affine_map<(d0) -> ()>
func @lg_nested_loop(%A: tensor<{{n}}x{{d}}xf64>, %B: tensor<{{k}}x{{d}}xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ck = arith.constant {{k}} : index
  %cn = arith.constant {{n}} : index
  %zero = arith.constant 0.0 : f64
  %mt_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>
  %mt_init = linalg.fill(%zero, %mt_space) : f64, tensor<{{k}}xf64> -> tensor<{{k}}xf64>
  %sumexp_space = arith.constant dense<0.0> : tensor<f64>
  %zerod_space = arith.constant dense<0.0> : tensor<f64>
  %final = scf.for %iv = %c0 to %cn step %c1 iter_args(%final_iv = %zero) -> f64 {
    %A_slice = tensor.extract_slice %A[%iv, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
    %main_term = scf.for %jv = %c0 to %ck step %c1 iter_args(%mt_iter = %mt_init) -> tensor<{{k}}xf64> {
      %B_slice = tensor.extract_slice %B[%jv, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
      %dotted = linalg.dot ins(%A_slice, %B_slice : tensor<{{d}}xf64>, tensor<{{d}}xf64>) outs(%zerod_space : tensor<f64>) -> tensor<f64>
      %dval = tensor.extract %dotted[] : tensor<f64>
      %mt_next = tensor.insert %dval into %mt_iter[%jv] : tensor<{{k}}xf64>
      scf.yield %mt_next : tensor<{{k}}xf64>
    }

    // logsumexp
    %sumexp = linalg.generic
      {
        indexing_maps = [#id_1d, #reduce_1d], iterator_types = ["reduction"]
      }
      ins(%main_term : tensor<{{k}}xf64>)
      outs(%sumexp_space : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = math.exp %arg0 : f64
      %1 = arith.addf %0, %arg1 : f64
      linalg.yield %1 : f64
    } -> tensor<f64>
    %sumexp_v = tensor.extract %sumexp[] : tensor<f64>
    %lse = math.log %sumexp_v : f64
    %final_next = arith.addf %lse, %final_iv : f64
    scf.yield %final_next : f64
  }
  return %final : f64
}

func @lg_main_term(%A: tensor<{{n}}x{{d}}xf64>, %B: tensor<{{k}}x{{d}}xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ck = arith.constant {{k}} : index
  %zero = arith.constant 0.0 : f64
  %mt_space = linalg.init_tensor [{{k}}] : tensor<{{k}}xf64>
  %mt_init = linalg.fill(%zero, %mt_space) : f64, tensor<{{k}}xf64> -> tensor<{{k}}xf64>
  %zerod_space = arith.constant dense<0.0> : tensor<f64>
  %main_term = scf.for %jv = %c0 to %ck step %c1 iter_args(%mt_iter = %mt_init) -> tensor<{{k}}xf64> {
    %A_slice = tensor.extract_slice %A[%jv, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
    %B_slice = tensor.extract_slice %B[%jv, 0] [1, {{d}}] [1, 1] : tensor<{{k}}x{{d}}xf64> to tensor<{{d}}xf64>
    %dotted = linalg.dot ins(%A_slice, %B_slice : tensor<{{d}}xf64>, tensor<{{d}}xf64>) outs(%zerod_space : tensor<f64>) -> tensor<f64>
    %dval = tensor.extract %dotted[] : tensor<f64>
    %mt_next = tensor.insert %dval into %mt_iter[%jv] : tensor<{{k}}xf64>
    scf.yield %mt_next : tensor<{{k}}xf64>
  }

  %sumexp = linalg.generic
    {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    }
    ins(%main_term : tensor<{{k}}xf64>)
    outs(%zerod_space : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = math.exp %arg0 : f64
    %1 = arith.addf %0, %arg1 : f64
    linalg.yield %1 : f64
  } -> tensor<f64>
  %sumexp_v = tensor.extract %sumexp[] : tensor<f64>
  %lse = math.log %sumexp_v : f64
  return %lse : f64
}

func @lagrad_main_term(%A: tensor<{{n}}x{{d}}xf64>, %B: tensor<{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{d}}xf64> {
  %f = constant @lg_main_term: (tensor<{{n}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>) -> f64
  %df = standalone.grad %f {of = [0]}: (tensor<{{n}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>) -> f64, (tensor<{{n}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{d}}xf64>
  %res = call_indirect %df(%A, %B) : (tensor<{{n}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{d}}xf64>
  return %res : tensor<{{n}}x{{d}}xf64>
}

func @lg_loop(%A: tensor<{{n}}x{{d}}xf64>) -> f64 {
  %zero = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cn = arith.constant {{n}} : index
  %r_init = arith.constant dense<0.0> : tensor<f64>
  %res = scf.for %iv = %c0 to %cn step %c1 iter_args(%sum_it = %zero) -> f64 {
    %slice = tensor.extract_slice %A[%iv, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
    %reduce = linalg.generic
      {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]
      }
      ins(%slice : tensor<{{d}}xf64>)
      outs(%r_init : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.addf %arg0, %arg1 : f64
      linalg.yield %0 : f64
    } -> tensor<f64>
    %r_el = tensor.extract %reduce[] : tensor<f64>
    %sum_next = arith.addf %r_el, %sum_it : f64
    scf.yield %sum_next : f64
  }
  return %res : f64
}

func @lagrad_nested_loop(%A: tensor<{{n}}x{{d}}xf64>, %B: tensor<{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{d}}xf64> {
  %f = constant @lg_nested_loop : (tensor<{{n}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>) -> f64
  %df = standalone.grad %f {of = [0]} : (tensor<{{n}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>) -> f64, (tensor<{{n}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{d}}xf64>
  %res = call_indirect %df(%A, %B) : (tensor<{{n}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{d}}xf64>
  return %res : tensor<{{n}}x{{d}}xf64>
}

func @lagrad_loop(%A: tensor<{{n}}x{{d}}xf64>) -> tensor<{{n}}x{{d}}xf64> {
  %f = constant @lg_loop : (tensor<{{n}}x{{d}}xf64>) -> f64
  %df = standalone.grad %f {of = [0]} : (tensor<{{n}}x{{d}}xf64>) -> f64, (tensor<{{n}}x{{d}}xf64>) -> tensor<{{n}}x{{d}}xf64>
  %res = call_indirect %df(%A) : (tensor<{{n}}x{{d}}xf64>) -> tensor<{{n}}x{{d}}xf64>
  return %res : tensor<{{n}}x{{d}}xf64>
}
