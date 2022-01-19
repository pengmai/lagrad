func @mygrad_rels_to_abs(%relatives: tensor<22x4x4xf64>, %parents: tensor<22xi32>) -> tensor<22x4x4xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n_one = arith.constant -1 : i32
  %cb = arith.constant 22 : index
  %matmul_init = arith.constant dense<0.0> : tensor<4x4xf64>
  %absolute_space = linalg.init_tensor [22, 4, 4] : tensor<22x4x4xf64>
  %absolutes = scf.for %iv = %c0 to %cb step %c1 iter_args(%a_iter = %absolute_space) -> (tensor<22x4x4xf64>) {
    %parent_i = tensor.extract %parents[%iv] : tensor<22xi32>
    %rel_i = tensor.extract_slice %relatives[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
    %pred = arith.cmpi "eq", %parent_i, %n_one : i32
    %result = scf.if %pred -> tensor<4x4xf64> {
      scf.yield %rel_i : tensor<4x4xf64>
    } else {
      %parent_idx = arith.index_cast %parent_i : i32 to index
      %abs_p = tensor.extract_slice %a_iter[%parent_idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
      // This is the ADBench orientation, not the Enzyme orientation.
      %abs_i = linalg.matmul ins(%abs_p, %rel_i : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) -> tensor<4x4xf64>
      scf.yield %abs_i : tensor<4x4xf64>
    }
    %a_next = tensor.insert_slice %result into %a_iter[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4xf64> into tensor<22x4x4xf64>
    scf.yield %a_next : tensor<22x4x4xf64>
  }

  %one = arith.constant 1.0 : f64
  %zero = arith.constant 0.0 : f64
  %last = arith.constant 21 : index
  %dabsolutes_space = linalg.init_tensor [22, 4, 4] : tensor<22x4x4xf64>
  %dabsolutes_init = linalg.fill(%one, %dabsolutes_space) : f64, tensor<22x4x4xf64> -> tensor<22x4x4xf64>
  %drelatives_space = linalg.init_tensor [22, 4, 4] : tensor<22x4x4xf64>
  %drelatives_init = linalg.fill(%zero, %drelatives_space) : f64, tensor<22x4x4xf64> -> tensor<22x4x4xf64>
  %dres:2 = scf.for %iv = %c0 to %cb step %c1 iter_args(%dabs = %dabsolutes_init, %drel = %drelatives_init) -> (tensor<22x4x4xf64>, tensor<22x4x4xf64>) {
    %idx = arith.subi %last, %iv : index
    %da_next = tensor.extract_slice %dabs[%idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
    %parent_i = tensor.extract %parents[%idx] : tensor<22xi32>
    %rel_i = tensor.extract_slice %relatives[%idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
    %pred = arith.cmpi "eq", %parent_i, %n_one : i32
    %dresult:2 = scf.if %pred -> (tensor<4x4xf64>, tensor<22x4x4xf64>) {
      scf.yield %da_next, %dabs : tensor<4x4xf64>, tensor<22x4x4xf64>
    } else {
      %parent_idx = arith.index_cast %parent_i : i32 to index
      %abs_p = tensor.extract_slice %absolutes[%parent_idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
      %dabs_i_wrt_abs_p = linalg.generic #matmul_adjoint_arg0 ins(%da_next, %rel_i : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        %1 = arith.addf %0, %arg2 : f64
        linalg.yield %1 :  f64
      } -> tensor<4x4xf64>
      %dabs_i_wrt_rel_i = linalg.generic #matmul_adjoint_arg1 ins(%abs_p, %da_next : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) {
      ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
        %0 = arith.mulf %arg0, %arg1 : f64
        %1 = arith.addf %0, %arg2 : f64
        linalg.yield %1 :  f64
      } -> tensor<4x4xf64>

      %dabs_next_0 = tensor.extract_slice %dabs[%parent_idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
      %dabs_next_1 = arith.addf %dabs_next_0, %dabs_i_wrt_abs_p : tensor<4x4xf64>
      %dabs_next = tensor.insert_slice %dabs_next_1 into %dabs[%parent_idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4xf64> into tensor<22x4x4xf64>
      scf.yield %dabs_i_wrt_rel_i, %dabs_next : tensor<4x4xf64>, tensor<22x4x4xf64>
    }
    // This is mainly for efficiency reasons
    %drel_slice_0 = tensor.extract_slice %drel[%idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
    %drel_slice_1 = arith.addf %drel_slice_0, %dresult#0 : tensor<4x4xf64>
    %drel_next = tensor.insert_slice %drel_slice_1 into %drel[%idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4xf64> into tensor<22x4x4xf64>
    scf.yield %dresult#1, %drel_next : tensor<22x4x4xf64>, tensor<22x4x4xf64>
  }
  return %dres#1 : tensor<22x4x4xf64>
}