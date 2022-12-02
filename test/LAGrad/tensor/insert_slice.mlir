// Be mindful of moving the entries of results around! This is currently broken.
func.func @insert(%t: tensor<4xf64>) -> tensor<5x4xf64> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %two = arith.constant 2.0 : f64
  %space = arith.constant dense<0.0> : tensor<5x4xf64>
  %res_0 = tensor.insert_slice %t into %space[%c3, 0] [1, 4] [1, 1] : tensor<4xf64> into tensor<5x4xf64>
  return %res_0 : tensor<5x4xf64>
}

func.func @disabled_insert(%t: tensor<4xf64>) -> tensor<5x4xf64> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %two = arith.constant 2.0 : f64
  %space = arith.constant dense<0.0> : tensor<5x4xf64>
  %res_0 = tensor.insert_slice %t into %space[%c3, 0] [1, 4] [1, 1] : tensor<4xf64> into tensor<5x4xf64>
  %th = tensor.extract %res_0[%c3, %c3] : tensor<5x4xf64>
  %th_1 = arith.mulf %th, %two : f64
  %res_1 = tensor.insert %th_1 into %res_0[%c3, %c3] : tensor<5x4xf64>
  return %res_1 : tensor<5x4xf64>
}

// func @dinsert(%t: tensor<4xf64>) -> tensor<4xf64> {
//   %c3 = arith.constant 3 : index
//   %zero = arith.constant 0.0 : f64
//   %two = arith.constant 2.0 : f64
//   %g = arith.constant dense<1.0> : tensor<5x4xf64>
//   %dres_1 = tensor.extract %g[%c3, %c3] : tensor<5x4xf64>
//   %dth_1 = arith.mulf %dres_1, %two : f64

//   // reverse tensor.extract
//   %space = linalg.init_tensor [5, 4] : tensor<5x4xf64>
//   %init = linalg.fill(%zero, %space) : f64, tensor<5x4xf64> -> tensor<5x4xf64>
//   %dth = tensor.insert %dth_1 into %init[%c3, %c3] : tensor<5x4xf64>

//   %dres_0 = tensor.extract_slice %dth[%c3, 0] [1, 4] [1, 1] : tensor<5x4xf64> to tensor<4xf64>
//   return %dres_0 : tensor<4xf64>
// }

func.func private @printMemrefF64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func.func @main() {
  %arg = arith.constant dense<[1., 2., 3., 4.]> : tensor<4xf64>
  %res = lagrad.grad @insert(%arg) : (tensor<4xf64>) -> tensor<4xf64>
  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  return
}
