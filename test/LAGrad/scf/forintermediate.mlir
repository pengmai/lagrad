func @forint(%A: tensor<2x4xf64>) -> tensor<2x4xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %t_init = linalg.init_tensor [2, 4] : tensor<2x4xf64>
  %res = scf.for %iv = %c0 to %c2 step %c1 iter_args(%t = %t_init) -> tensor<2x4xf64> {
    %A_slice = tensor.extract_slice %A[%iv, 0] [1, 4] [1, 1] : tensor<2x4xf64> to tensor<4xf64>
    %pred = arith.cmpi "eq", %iv, %c0 : index
    %t_slice = scf.if %pred -> tensor<4xf64> {
      scf.yield %A_slice : tensor<4xf64>
    } else {
      %t_prev = tensor.extract_slice %t[0, 0] [1, 4] [1, 1] : tensor<2x4xf64> to tensor<4xf64>
      %t_0 = arith.mulf %t_prev, %A_slice : tensor<4xf64>
      scf.yield %t_0 : tensor<4xf64>
    }
    %t_next = tensor.insert_slice %t_slice into %t[%iv, 0] [1, 4] [1, 1] : tensor<4xf64> into tensor<2x4xf64>
    scf.yield %t_next : tensor<2x4xf64>
  }
  return %res : tensor<2x4xf64>
}

func @mygrad_forint(%A: tensor<2x4xf64>) -> tensor<2x4xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f64
  %t_init = linalg.init_tensor [2, 4] : tensor<2x4xf64>
  // %t_init = linalg.fill(%zero, %t_space) : f64, tensor<2x4xf64> -> tensor<2x4xf64>
  %stack_init = linalg.init_tensor [2, 2, 4] : tensor<2x2x4xf64>
  %res:2 = scf.for %iv = %c0 to %c2 step %c1 iter_args(%t = %t_init, %stack_it = %stack_init) -> (tensor<2x4xf64>, tensor<2x2x4xf64>) {
    %A_slice = tensor.extract_slice %A[%iv, 0] [1, 4] [1, 1] : tensor<2x4xf64> to tensor<4xf64>
    %pred = arith.cmpi "eq", %iv, %c0 : index
    %t_slice = scf.if %pred -> tensor<4xf64> {
      scf.yield %A_slice : tensor<4xf64>
    } else {
      %t_prev = tensor.extract_slice %t[0, 0] [1, 4] [1, 1] : tensor<2x4xf64> to tensor<4xf64>
      %t_0 = arith.mulf %t_prev, %A_slice : tensor<4xf64>
      scf.yield %t_0 : tensor<4xf64>
    }
    %t_next = tensor.insert_slice %t_slice into %t[%iv, 0] [1, 4] [1, 1] : tensor<4xf64> into tensor<2x4xf64>

    // Augment the primal with stack values
    %stack_next = tensor.insert_slice %t_next into %stack_it[%iv, 0, 0] [1, 2, 4] [1, 1, 1] : tensor<2x4xf64> into tensor<2x2x4xf64>
    scf.yield %t_next, %stack_next : tensor<2x4xf64>, tensor<2x2x4xf64>
  }

  // Iterate in reverse
  %g_init = arith.constant dense<1.0> : tensor<2x4xf64>
  %dA_space = linalg.init_tensor [2, 4] : tensor<2x4xf64>
  %dA_init = linalg.fill(%zero, %dA_space) : f64, tensor<2x4xf64> -> tensor<2x4xf64>
  %dres:2 = scf.for %iv = %c0 to %c2 step %c1 iter_args(%dt = %g_init, %dA = %dA_init) -> (tensor<2x4xf64>, tensor<2x4xf64>) {
    // Required to compute the primal %iv
    %idx_0 = arith.subi %c2, %iv : index
    %idx = arith.subi %idx_0, %c1 : index
    // %t = tensor.extract_slice %res#1[%idx, 0, 0] [1, 2, 4] [1, 1, 1] : tensor<2x2x4xf64> to tensor<2x4xf64>

    %A_slice = tensor.extract_slice %A[%idx, 0] [1, 4] [1, 1] : tensor<2x4xf64> to tensor<4xf64>
    // Differentiate insert_slice
    %dt_next = tensor.extract_slice %dt[%idx, 0] [1, 4] [1, 1] : tensor<2x4xf64> to tensor<4xf64>
    // Careful with the induction variable; we have to iterate here in reverse
    %pred = arith.cmpi "eq", %idx, %c0 : index
    // Differentiate with respect to %A_slice
    %dt_slice:2 = scf.if %pred -> (tensor<4xf64>, tensor<2x4xf64>) {
      %zerot = arith.constant dense<0.0> : tensor<2x4xf64>
      scf.yield %dt_next, %zerot : tensor<4xf64>, tensor<2x4xf64>
    } else {
      %t_prev = tensor.extract_slice %res#0[0, 0] [1, 4] [1, 1] : tensor<2x4xf64> to tensor<4xf64>
      %dt_0 = arith.mulf %t_prev, %dt_next : tensor<4xf64>
      %dt_0_wrt_t = arith.mulf %A_slice, %dt_next : tensor<4xf64>
      %dt_prev = tensor.insert_slice %dt_0_wrt_t into %dt[0, 0] [1, 4] [1, 1] : tensor<4xf64> into tensor<2x4xf64>
      scf.yield %dt_0, %dt_prev : tensor<4xf64>, tensor<2x4xf64>
    }

    // %tmp = linalg.init_tensor [2, 4] : tensor<2x4xf64>
    // %tmp_1 = linalg.fill(%zero, %tmp) : f64, tensor<2x4xf64> -> tensor<2x4xf64>
    %dA_next = tensor.insert_slice %dt_slice into %dA[%idx, 0] [1, 4] [1, 1] : tensor<4xf64> into tensor<2x4xf64>
    // %dA_next = arith.addf %tmp_2, %dA : tensor<2x4xf64>
    scf.yield %dt_slice#1, %dA_next : tensor<2x4xf64>, tensor<2x4xf64>
  }
  return %dres#1 : tensor<2x4xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %arg = arith.constant dense<[
    [1., 2., 3., 4.],
    [5., 6., 7., 8.]
  ]> : tensor<2x4xf64>
  // %f = constant @forint : (tensor<2x4xf64>) -> tensor<2x4xf64>
  // %df = standalone.grad %f : (tensor<2x4xf64>) -> tensor<2x4xf64>, (tensor<2x4xf64>) -> tensor<2x4xf64>
  // %res = call_indirect %df(%arg) : (tensor<2x4xf64>) -> tensor<2x4xf64>
  %res = call @mygrad_forint(%arg) : (tensor<2x4xf64>) -> tensor<2x4xf64>
  %U = tensor.cast %res : tensor<2x4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
