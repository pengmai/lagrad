func @insert_overwrite(%t: tensor<2x2xf64>) -> tensor<2x2xf64> {
  %cst = arith.constant dense<2.0> : tensor<1x2xf64>
  %s = tensor.extract_slice %t[1, 0] [1, 2] [1, 1] : tensor<2x2xf64> to tensor<1x2xf64>
  %m = arith.mulf %s, %cst : tensor<1x2xf64>
  %t_new = tensor.insert_slice %m into %t[1, 0] [1, 2] [1, 1] : tensor<1x2xf64> into tensor<2x2xf64>
  return %t_new : tensor<2x2xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %t = arith.constant dense<-1.0> : tensor<2x2xf64>
  %f = constant @insert_overwrite : (tensor<2x2xf64>) -> tensor<2x2xf64>
  %df = standalone.grad %f : (tensor<2x2xf64>) -> tensor<2x2xf64>, (tensor<2x2xf64>) -> tensor<2x2xf64>
  %res = call_indirect %df(%t) : (tensor<2x2xf64>) -> tensor<2x2xf64>
  // %res = call @mygrad_insert_overwrite(%t) : (tensor<2x2xf64>) -> tensor<2x2xf64>
  %U = tensor.cast %res : tensor<2x2xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}

// func @mygrad_insert_overwrite(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
//   %cst = arith.constant 1.000000e+00 : f64
//   %cst_0 = arith.constant dense<2.000000e+00> : tensor<2xf64>
//   %zerot = arith.constant dense<0.0> : tensor<2xf64>
//   // g is initialized here
//   %0 = linalg.init_tensor [2, 2] : tensor<2x2xf64>
//   %1 = linalg.fill(%cst, %0) : f64, tensor<2x2xf64> -> tensor<2x2xf64>

//   // gradient of tensor.insert_slice
//   %2 = tensor.extract_slice %1[1, 0] [1, 2] [1, 1] : tensor<2x2xf64> to tensor<2xf64>
//   %new_1 = tensor.insert_slice %zerot into %1[1, 0] [1, 2] [1, 1] : tensor<2xf64> into tensor<2x2xf64>

//   // gradient of mul
//   %3 = arith.mulf %2, %cst_0 : tensor<2xf64>

//   // gradient of tensor.extract_slice
//   %4 = tensor.extract_slice %new_1[1, 0] [1, 2] [1, 1] : tensor<2x2xf64> to tensor<2xf64>
//   // This add is incorrect. However, there are cases (like for loop adjoints)
//   // where this is the desired behaviour.
//   %5 = arith.addf %4, %3 : tensor<2xf64>
//   %6 = tensor.insert_slice %5 into %new_1[1, 0] [1, 2] [1, 1] : tensor<2xf64> into tensor<2x2xf64>
//   return %6 : tensor<2x2xf64>
// }
