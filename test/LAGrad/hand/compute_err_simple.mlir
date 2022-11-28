func @compute_err(%vertex_positions: tensor<544x3xf64>, %points: tensor<2x3xf64>, %correspondences: tensor<2xi32>) -> tensor<2x3xf64> { 
  %err_init = linalg.init_tensor [2, 3] : tensor<2x3xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cpts = arith.constant 2 : index
  %err = scf.for %iv = %c0 to %cpts step %c1 iter_args(%e_outer = %err_init) -> tensor<2x3xf64> {
    %err_partial = scf.for %jv = %c0 to %c3 step %c1 iter_args(%e_inner = %e_outer) -> tensor<2x3xf64> {
      %arg0 = tensor.extract %points[%iv, %jv] : tensor<2x3xf64>
      %i_0 = tensor.extract %correspondences[%iv] : tensor<2xi32>
      %i = arith.index_cast %i_0 : i32 to index
      %vp = tensor.extract %vertex_positions[%i, %jv] : tensor<544x3xf64>
      %0 = arith.subf %arg0, %vp : f64
      %e_next = tensor.insert %0 into %e_inner[%iv, %jv] : tensor<2x3xf64>
      scf.yield %e_next : tensor<2x3xf64>
    }
    scf.yield %err_partial : tensor<2x3xf64>
  }
  return %err : tensor<2x3xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %vertex_positions = tensor.generate {
  ^bb0(%i: index, %j: index):
    %0 = arith.muli %i, %c3 : index
    %1 = arith.addi %0, %j : index
    %2 = arith.addi %1, %c1 : index
    %3 = arith.index_cast %2 : index to i64
    %4 = arith.sitofp %3 : i64 to f64
    tensor.yield %4 : f64
  } : tensor<544x3xf64>
  %points = arith.constant dense<[[1., 2., 3.], [4., 5., 6.]]> : tensor<2x3xf64>
  %correspondences = arith.constant dense<[10, 16]> : tensor<2xi32>

  %adjoint = lagrad.grad @compute_err(%vertex_positions, %points, %correspondences) : (tensor<544x3xf64>, tensor<2x3xf64>, tensor<2xi32>) -> tensor<544x3xf64>
  %U = tensor.cast %adjoint : tensor<544x3xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
