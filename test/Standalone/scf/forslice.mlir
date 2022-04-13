func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @forslice(%A: tensor<4x5xf64>, %B: tensor<3x5xf64>) -> f64 {
  %zero = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %it_space = linalg.init_tensor [5] : tensor<5xf64>
  %zerod_tensor = arith.constant dense<0.0> : tensor<f64>
  %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%scal = %zero) -> f64 {
    %res_inner = scf.for %jv = %c0 to %c3 step %c1 iter_args(%it = %it_space) -> tensor<5xf64> {
      %Aslice = tensor.extract_slice %A[%iv, 0] [1, 5] [1, 1] : tensor<4x5xf64> to tensor<5xf64>
      %Bslice = tensor.extract_slice %B[%jv, 0] [1, 5] [1, 1] : tensor<3x5xf64> to tensor<5xf64>
      %inner_next = linalg.dot ins(%Aslice, %Bslice : tensor<5xf64>, tensor<5xf64>) outs(%zerod_tensor : tensor<f64>) -> tensor<f64>
      // %inner_next = linalg.generic
      //   {
      //     indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      //     iterator_types = ["reduction"]
      //   }
      //   ins(%Aslice : tensor<5xf64>)
      //   outs(%zerod_tensor : tensor<f64>) {
      // ^bb0(%arg0: f64, %arg1: f64):
      //   %0 = arith.mulf %arg0, %Bslice : f64
      //   %1 = arith.addf %0, %arg1 : f64
      //   linalg.yield %1 : f64
      // } -> tensor<f64>
      %inner_next_val = tensor.extract %inner_next[] : tensor<f64>
      %inner_inserted = tensor.insert %inner_next_val into %it[%jv] : tensor<5xf64>
      scf.yield %inner_inserted : tensor<5xf64>
    }
    %inner_summed = linalg.generic
      {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]
      }
      ins(%res_inner : tensor<5xf64>)
      outs(%zerod_tensor : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.addf %arg0, %arg1 : f64
      linalg.yield %0 : f64
    } -> tensor<f64>
    %inner_val = tensor.extract %inner_summed[] : tensor<f64>
    %it_next = arith.addf %inner_val, %scal : f64
    scf.yield %it_next : f64
  }
  return %res : f64
}

func @main() {
  %A = arith.constant dense<[
    [ 0.,  1.,  2.,  3.,  4.],
    [ 5.,  6.,  7.,  8.,  9.],
    [10., 11., 12., 13., 14.],
    [15., 16., 17., 18., 19.]
  ]> : tensor<4x5xf64>
  %B = arith.constant dense<[
    [ 0.,  1.,  2.,  3.,  4.],
    [ 5.,  6.,  7.,  8.,  9.],
    [10., 11., 12., 13., 14.]
  ]> : tensor<3x5xf64>
  %f = constant @forslice : (tensor<4x5xf64>, tensor<3x5xf64>) -> f64
  %df = standalone.grad %f {of = [0, 1]} : (tensor<4x5xf64>, tensor<3x5xf64>) -> f64, (tensor<4x5xf64>, tensor<3x5xf64>) -> (tensor<4x5xf64>, tensor<3x5xf64>)
  %res:2 = call_indirect %df(%A, %B) : (tensor<4x5xf64>, tensor<3x5xf64>) -> (tensor<4x5xf64>, tensor<3x5xf64>)
  %U_0 = tensor.cast %res#0 : tensor<4x5xf64> to tensor<*xf64>
  call @print_memref_f64(%U_0) : (tensor<*xf64>) -> ()
  %U1 = tensor.cast %res#1 : tensor<3x5xf64> to tensor<*xf64>
  call @print_memref_f64(%U1) : (tensor<*xf64>) -> ()
  return
}
