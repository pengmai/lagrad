func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @fortensor(%A : tensor<4x4xf64>, %b_space : tensor<4xf64>) -> tensor<4xf64> {
  %out = arith.constant dense<0.0> : tensor<f64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%t_iter = %b_space) -> (tensor<4xf64>) {
    %0 = linalg.generic
      {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>],
        iterator_types = ["reduction", "reduction"]
      }
      ins(%A : tensor<4x4xf64>)
      outs(%out : tensor<f64>) {
    ^bb0(%arg2: f64, %arg3: f64):
      %1 = arith.addf %arg2, %arg3 : f64
      linalg.yield %1 : f64
    } -> tensor<f64>
    %1 = tensor.extract %0[] : tensor<f64>
    %t_next = tensor.insert %1 into %t_iter[%iv] : tensor<4xf64>
    scf.yield %t_next : tensor<4xf64>
  }
  return %res : tensor<4xf64>
}

// func @mygrad_fortensor(%A : tensor<4x4xf64>, %b_space : tensor<5xf64>) -> tensor<4x4xf64> {
//   %zero = arith.constant 0.0 : f64
//   %out = arith.constant dense<0.0> : tensor<f64>
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %cn = arith.constant 5 : index
//   %grad_out = arith.constant dense<0.0> : tensor<4x4xf64>
//   %signal = arith.constant dense<1.0> : tensor<5xf64>

//   // Need to augment the for loop with both the gradient signal and gradient space?
//   // Maybe just the gradient space is enough. I think the signal stays the same every iteration.
//   %res = scf.for %iv = %c0 to %cn step %c1 iter_args(%g_iter = %grad_out) -> (tensor<4x4xf64>) {
//     // Differentiate tensor.insert
//     %g = tensor.extract %signal[%iv] : tensor<5xf64>
//     // Differentiate tensor.extract
//     %space = linalg.init_tensor [] : tensor<f64>
//     %filled = linalg.fill(%zero, %space) : f64, tensor<f64> -> tensor<f64>
//     %g1 = tensor.insert %g into %filled[] : tensor<f64>
//     // Differentiate through linalg.generic
//     %d0_space = arith.constant dense<0.0> : tensor<4x4xf64>
//     %d0 = linalg.generic
//       {
//         indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
//         iterator_types = ["parallel", "parallel"]
//       }
//       ins(%A, %g1 : tensor<4x4xf64>, tensor<f64>)
//       outs(%d0_space : tensor<4x4xf64>) {
//     ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
//       linalg.yield %arg1 : f64
//     } -> tensor<4x4xf64>
//     // In this case I need to add together the gradient signals,
//     // just like linalg.generic.
//     %grad_next = arith.addf %d0, %g_iter : tensor<4x4xf64>
//     scf.yield %grad_next : tensor<4x4xf64>
//   }
//   return %res : tensor<4x4xf64>
// }

func @main() {
  %A = arith.constant dense<1.1> : tensor<4x4xf64>
  %b_space = arith.constant dense<0.0> : tensor<4xf64>

  %f = constant @fortensor : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4xf64>
  %df = standalone.grad %f : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4xf64>, (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4x4xf64>
  %res = call_indirect %df(%A, %b_space) : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4x4xf64>
  %U = tensor.cast %res : tensor<4x4xf64> to tensor<*xf64>

  // %res = call @fortensor(%A, %b_space) : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4xf64>

  // %res = call @mygrad_fortensor(%A, %b_space) : (tensor<4x4xf64>, tensor<5xf64>) -> tensor<4x4xf64>
  // %U = tensor.cast %res : tensor<4x4xf64> to tensor<*xf64>

  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
