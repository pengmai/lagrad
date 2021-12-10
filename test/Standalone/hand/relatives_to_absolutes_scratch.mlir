// How would this look with respect to a theoretical scan op?
func @myscan() {
  %arg = arith.constant dense<[1., 2., 3., 4., 5.]> : tensor<5xf64>
  // This would produce [1, 3, 5, 9, 14]
  %res = standalone.scan ins(%arg : tensor<5xf64>) {
  ^bb0(%arg0: f64, %acc: f64):
    %0 = arith.addf %arg0, %acc : f64
    standalone.yield %0 : f64
  } -> f64
}

func @mrelatives_to_absolutes(%relatives: tensor<22x4x4xf64>, %parents: tensor<22xi32>) -> tensor<22x4x4xf64> {
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
  return %absolutes : tensor<22x4x4xf64>
}