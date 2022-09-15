func @onehot_adjoint_err_nest(%arg8: tensor<?x3xf64, "onehot">, %arg6: tensor<?xi32>) -> tensor<544x3xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %cst_0 = arith.constant 0.000000e+00 : f64
  %9 = tensor.dim %arg8, %c0 : tensor<?x3xf64, "onehot">
  %10 = linalg.init_tensor [544, 3] : tensor<544x3xf64>
  %11 = linalg.fill(%cst_0, %10) {"gradient space for %vertex_positions"} : f64, tensor<544x3xf64> -> tensor<544x3xf64>
  %12:2 = scf.for %arg9 = %c0 to %9 step %c1 iter_args(%arg10 = %arg8, %arg11 = %11) -> (tensor<?x3xf64, "onehot">, tensor<544x3xf64>) {
    %38 = arith.subi %9, %arg9 : index
    %39 = arith.subi %38, %c1 : index
    %40:2 = scf.for %arg12 = %c0 to %c3 step %c1 iter_args(%arg13 = %arg10, %arg14 = %arg11) -> (tensor<?x3xf64, "onehot">, tensor<544x3xf64>) {
      %41 = arith.subi %c2, %arg12 : index
      %42 = tensor.extract %arg6[%39] {"cloned %i_0"} : tensor<?xi32>
      %43 = arith.index_cast %42 {"cloned %i"} : i32 to index
      %44 = tensor.extract %arg13[%39, %41] : tensor<?x3xf64, "onehot">
      %45 = tensor.insert %cst_0 into %arg13[%39, %41] : tensor<?x3xf64, "onehot">
      %46 = arith.negf %44 : f64
      %47 = tensor.extract %arg14[%43, %41] : tensor<544x3xf64>
      %48 = arith.addf %47, %46 : f64
      %49 = tensor.insert %48 into %arg14[%43, %41] : tensor<544x3xf64>
      scf.yield %45, %49 : tensor<?x3xf64, "onehot">, tensor<544x3xf64>
    } {"adjoint of %err_partial"}
    scf.yield %40#0, %40#1 : tensor<?x3xf64, "onehot">, tensor<544x3xf64>
  } {"adjoint of %err"}
  return %12#1 : tensor<544x3xf64>
  // return %11 : tensor<544x3xf64>
}
