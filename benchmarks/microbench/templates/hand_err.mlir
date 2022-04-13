// From early testing it looks like this part of computing the adjoint is the most expensive?

func @la_hand_err(%vertex_positions: tensor<544x3xf64>, %points: tensor<100x3xf64>, %correspondences: tensor<100xi32>) -> tensor<100x3xf64> {
  %err_init = linalg.init_tensor [100, 3] : tensor<100x3xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cpts = arith.constant 100 : index
  %space = linalg.init_tensor [100, 3] : tensor<100x3xf64>
  %err = scf.for %iv = %c0 to %cpts step %c1 iter_args(%e_outer = %err_init) -> tensor<100x3xf64> {
    %err_partial = scf.for %jv = %c0 to %c3 step %c1 iter_args(%e_inner = %e_outer) -> tensor<100x3xf64> {
      %arg0 = tensor.extract %points[%iv, %jv] : tensor<100x3xf64>
      %i_0 = tensor.extract %correspondences[%iv] : tensor<100xi32>
      %i = arith.index_cast %i_0 : i32 to index
      %vp = tensor.extract %vertex_positions[%i, %jv] : tensor<544x3xf64>
      %0 = arith.subf %arg0, %vp : f64
      %e_next = tensor.insert %0 into %e_inner[%iv, %jv] : tensor<100x3xf64>
      scf.yield %e_next : tensor<100x3xf64>
    }
    scf.yield %err_partial : tensor<100x3xf64>
  }
  return %err : tensor<100x3xf64>
}

func @mygrad_la_hand_err(%arg0: tensor<544x3xf64>, %arg1: tensor<100x3xf64>, %arg2: tensor<100xi32>) -> tensor<544x3xf64> {
  %c2 = arith.constant 2 : index
  %c99 = arith.constant 99 : index
  %cst = arith.constant 0.000000e+00 : f64
  %cst_0 = arith.constant 1.000000e+00 : f64
  %c100 = arith.constant 100 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = linalg.init_tensor [100, 3] : tensor<100x3xf64>
  %1 = linalg.fill(%cst_0, %0) : f64, tensor<100x3xf64> -> tensor<100x3xf64>
  %2 = linalg.init_tensor [544, 3] : tensor<544x3xf64>
  %3 = linalg.fill(%cst, %2) : f64, tensor<544x3xf64> -> tensor<544x3xf64>
  %4 = scf.for %arg3 = %c0 to %c100 step %c1 iter_args(%arg4 = %3) -> (tensor<544x3xf64>) {
    %5 = arith.subi %c99, %arg3 : index
    %6 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<544x3xf64>) {
      %8 = arith.subi %c2, %arg5 : index
      %9 = tensor.extract %arg2[%5] : tensor<100xi32>
      %10 = arith.index_cast %9 : i32 to index
      %11 = tensor.extract %1[%5, %8] : tensor<100x3xf64>
      %12 = arith.negf %11 : f64
      %13 = tensor.extract %arg6[%10, %8] : tensor<544x3xf64>
      %14 = arith.addf %13, %12 : f64
      %15 = tensor.insert %14 into %arg6[%10, %8] : tensor<544x3xf64>
      scf.yield %15 : tensor<544x3xf64>
    }
    // %7 = arith.addf %arg4, %6 : tensor<544x3xf64>
    scf.yield %6 : tensor<544x3xf64>
  }
  return %4 : tensor<544x3xf64>
}

func @dla_hand_err(%vertex_positions: tensor<544x3xf64>, %points: tensor<100x3xf64>, %correspondences: tensor<100xi32>) -> tensor<544x3xf64> {
  %f = constant @la_hand_err : (tensor<544x3xf64>, tensor<100x3xf64>, tensor<100xi32>) -> tensor<100x3xf64>
  // %df = standalone.grad %f {of = [0]} : (tensor<544x3xf64>, tensor<100x3xf64>, tensor<100xi32>) -> tensor<100x3xf64>, (tensor<544x3xf64>, tensor<100x3xf64>, tensor<100xi32>) -> tensor<544x3xf64>
  // %res = call_indirect %df(%vertex_positions, %points, %correspondences) : (tensor<544x3xf64>, tensor<100x3xf64>, tensor<100xi32>) -> tensor<544x3xf64>
  %res = call @mygrad_la_hand_err(%vertex_positions, %points, %correspondences) : (tensor<544x3xf64>, tensor<100x3xf64>, tensor<100xi32>) -> tensor<544x3xf64>
  return %res : tensor<544x3xf64>
}
