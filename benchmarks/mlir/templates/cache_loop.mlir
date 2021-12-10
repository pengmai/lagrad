func @cache_loop(%A: tensor<{{n}}x{{d}}xf64>, %parents: tensor<{{n}}xi32>) -> tensor<{{n}}x{{d}}xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cneg1 = arith.constant -1 : index
  %cn = arith.constant {{n}} : index
  %res_init = linalg.init_tensor [{{n}}, {{d}}] : tensor<{{n}}x{{d}}xf64>
  %res = scf.for %iv = %c0 to %cn step %c1 iter_args(%res_it = %res_init) -> tensor<{{n}}x{{d}}xf64> {
    %parent_i = tensor.extract %parents[%iv] : tensor<{{n}}xi32>
    %parent_idx = arith.index_cast %parent_i : i32 to index
    %pred = arith.cmpi "eq", %parent_idx, %cneg1 : index
    %A_slice = tensor.extract_slice %A[%iv, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
    %row_it = scf.if %pred -> tensor<{{d}}xf64> {
      scf.yield %A_slice : tensor<{{d}}xf64>
    } else {
      %parent_row = tensor.extract_slice %res_it[%parent_idx, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
      %0 = arith.mulf %parent_row, %A_slice : tensor<{{d}}xf64>
      scf.yield %0 : tensor<{{d}}xf64>
    }
    %res_next = tensor.insert_slice %row_it into %res_it[%iv, 0] [1, {{d}}] [1, 1] : tensor<{{d}}xf64> into tensor<{{n}}x{{d}}xf64>
    scf.yield %res_next : tensor<{{n}}x{{d}}xf64>
  }
  return %res : tensor<{{n}}x{{d}}xf64>
}

// func @lagrad_cache_loop(%A: tensor<{{n}}x{{d}}xf64>) -> tensor<{{n}}x{{d}}xf64> {}

func @manual_cache_loop(%A: tensor<{{n}}x{{d}}xf64>, %parents: tensor<{{n}}xi32>) -> tensor<{{n}}x{{d}}xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cneg1 = arith.constant -1 : index
  %cn = arith.constant {{n}} : index
  %res_init = linalg.init_tensor [{{n}}, {{d}}] : tensor<{{n}}x{{d}}xf64>
  %res = scf.for %iv = %c0 to %cn step %c1 iter_args(%res_it = %res_init) -> (tensor<{{n}}x{{d}}xf64>) {
    %parent_i = tensor.extract %parents[%iv] : tensor<{{n}}xi32>
    %parent_idx = arith.index_cast %parent_i : i32 to index
    %pred = arith.cmpi "eq", %parent_idx, %cneg1 : index
    %A_slice = tensor.extract_slice %A[%iv, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
    %row_it = scf.if %pred -> tensor<{{d}}xf64> {
      scf.yield %A_slice : tensor<{{d}}xf64>
    } else {
      %parent_row = tensor.extract_slice %res_it[%parent_idx, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
      %0 = arith.mulf %parent_row, %A_slice : tensor<{{d}}xf64>
      scf.yield %0 : tensor<{{d}}xf64>
    }
    %res_next = tensor.insert_slice %row_it into %res_it[%iv, 0] [1, {{d}}] [1, 1] : tensor<{{d}}xf64> into tensor<{{n}}x{{d}}xf64>
    scf.yield %res_next : tensor<{{n}}x{{d}}xf64>
  }

  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %zerot = arith.constant dense<0.0> : tensor<{{d}}xf64>
  %dres_space = linalg.init_tensor [{{n}}, {{d}}] : tensor<{{n}}x{{d}}xf64>
  %dres = linalg.fill(%one, %dres_space) : f64, tensor<{{n}}x{{d}}xf64> -> tensor<{{n}}x{{d}}xf64>
  %dA_space = linalg.init_tensor [{{n}}, {{d}}] : tensor<{{n}}x{{d}}xf64>
  %dA_init = linalg.fill(%zero, %dA_space) : f64, tensor<{{n}}x{{d}}xf64> -> tensor<{{n}}x{{d}}xf64> 
  %dloop:2 = scf.for %iv = %c0 to %cn step %c1 iter_args(%dres_it = %dres, %dA_it = %dA_init) -> (tensor<{{n}}x{{d}}xf64>, tensor<{{n}}x{{d}}xf64>) {
    %idx_0 = arith.subi %cn, %iv : index
    %idx = arith.subi %idx_0, %c1 : index

    %dres_next = tensor.extract_slice %dres_it[%idx, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
    %A_slice = tensor.extract_slice %A[%idx, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
    %parent_i = tensor.extract %parents[%idx] : tensor<{{n}}xi32>
    %parent_idx = arith.index_cast %parent_i : i32 to index
    %pred = arith.cmpi "eq", %parent_idx, %cneg1 : index
    %drow_it:2 = scf.if %pred -> (tensor<{{d}}xf64>, tensor<{{n}}x{{d}}xf64>) {
      scf.yield %dres_next, %dres_it : tensor<{{d}}xf64>, tensor<{{n}}x{{d}}xf64>
    } else {
      %parent_row = tensor.extract_slice %res[%parent_idx, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
      %dparent_row = tensor.extract_slice %dres_it[%parent_idx, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
      %d0_wrt_parent_row = arith.mulf %A_slice, %dres_next : tensor<{{d}}xf64>
      %d0_wrt_A_slice = arith.mulf %parent_row, %dres_next : tensor<{{d}}xf64>

      %dparent_row_next = arith.addf %dparent_row, %d0_wrt_parent_row : tensor<{{d}}xf64>
      %dres_0 = tensor.insert_slice %dparent_row_next into %dres_it[%parent_idx, 0] [1, {{d}}] [1, 1] : tensor<{{d}}xf64> into tensor<{{n}}x{{d}}xf64>

      scf.yield %d0_wrt_A_slice, %dres_0 : tensor<{{d}}xf64>, tensor<{{n}}x{{d}}xf64>
    }
    %dA_it_slice = tensor.extract_slice %dA_it[%idx, 0] [1, {{d}}] [1, 1] : tensor<{{n}}x{{d}}xf64> to tensor<{{d}}xf64>
    %dA_it_slice_next = arith.addf %dA_it_slice, %drow_it#0 : tensor<{{d}}xf64>
    %dA_next = tensor.insert_slice %dA_it_slice_next into %dA_it[%idx, 0] [1, {{d}}] [1, 1] : tensor<{{d}}xf64> into tensor<{{n}}x{{d}}xf64>
    scf.yield %drow_it#1, %dA_next : tensor<{{n}}x{{d}}xf64>, tensor<{{n}}x{{d}}xf64>
  }
  return %dloop#1 : tensor<{{n}}x{{d}}xf64>
}
