// This is to examine a bug with inserting and overwriting in a loop.

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %one = arith.constant 1.0 : f64
  %zero4 = arith.constant dense<0.0> : tensor<4xf64>
  %g_space = linalg.init_tensor [3, 4] : tensor<3x4xf64>
  %g_init = linalg.fill(%one, %g_space) : f64, tensor<3x4xf64> -> tensor<3x4xf64>
  %res:2 = scf.for %iv = %c0 to %c3 step %c1 iter_args(%res_i = %zero4, %g = %g_init) -> (tensor<4xf64>, tensor<3x4xf64>) {
    %gslice = tensor.extract_slice %g[%iv, 0] [1, 4] [1, 1] : tensor<3x4xf64> to tensor<4xf64>
    %gnext = tensor.insert_slice %zero4 into %g[%iv, 0] [1, 4] [1, 1] : tensor<4xf64> into tensor<3x4xf64>
    %res_next = arith.addf %gslice, %res_i : tensor<4xf64>
    scf.yield %res_next, %gnext : tensor<4xf64>, tensor<3x4xf64>
  }
  %U = tensor.cast %res#0 : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  %U2 = tensor.cast %res#1 : tensor<3x4xf64> to tensor<*xf64>
  call @print_memref_f64(%U2) : (tensor<*xf64>) -> ()
  return
}
