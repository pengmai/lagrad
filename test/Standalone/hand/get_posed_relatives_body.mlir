func.func @meuler_angles_to_rotation_matrix(%xyz: tensor<3xf64>) -> tensor<3x3xf64> attributes {pure = true} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %tx = tensor.extract %xyz[%c0] : tensor<3xf64>
  %ty = tensor.extract %xyz[%c2] : tensor<3xf64>
  %tz = tensor.extract %xyz[%c1] : tensor<3xf64>

  %Rx_0 = tensor.empty() : tensor<3x3xf64>
  %Rx_1 = linalg.fill ins(%zero: f64) outs(%Rx_0: tensor<3x3xf64>) -> tensor<3x3xf64>
  %cos_tx = math.cos %tx : f64
  %Rx_2 = tensor.insert %cos_tx into %Rx_1[%c1, %c1] : tensor<3x3xf64>
  %sin_tx = math.sin %tx : f64
  %Rx_3 = tensor.insert %sin_tx into %Rx_2[%c1, %c2] : tensor<3x3xf64>
  %nsin_tx = arith.negf %sin_tx : f64
  %Rx_4 = tensor.insert %nsin_tx into %Rx_3[%c2, %c1] : tensor<3x3xf64>
  %Rx_5 = tensor.insert %cos_tx into %Rx_4[%c2, %c2] : tensor<3x3xf64>
  %Rx = tensor.insert %one into %Rx_5[%c0, %c0] : tensor<3x3xf64>

  %Ry_0 = tensor.empty() : tensor<3x3xf64>
  %Ry_1 = linalg.fill ins(%zero: f64) outs(%Ry_0: tensor<3x3xf64>) -> tensor<3x3xf64>
  %cos_ty = math.cos %ty : f64
  %Ry_2 = tensor.insert %cos_ty into %Ry_1[%c0, %c0] : tensor<3x3xf64>
  %sin_ty = math.sin %ty : f64
  %Ry_3 = tensor.insert %sin_ty into %Ry_2[%c0, %c2] : tensor<3x3xf64>
  %nsin_ty = arith.negf %sin_ty : f64
  %Ry_4 = tensor.insert %nsin_ty into %Ry_3[%c2, %c0] : tensor<3x3xf64>
  %Ry_5 = tensor.insert %cos_ty into %Ry_4[%c2, %c2] : tensor<3x3xf64>
  %Ry = tensor.insert %one into %Ry_5[%c1, %c1] : tensor<3x3xf64>

  %Rz_0 = tensor.empty() : tensor<3x3xf64>
  %Rz_1 = linalg.fill ins(%zero: f64) outs(%Rz_0: tensor<3x3xf64>) -> tensor<3x3xf64>
  %cos_tz = math.cos %tz : f64
  %Rz_2 = tensor.insert %cos_tz into %Rz_1[%c0, %c0] : tensor<3x3xf64>
  %sin_tz = math.sin %tz : f64
  %Rz_3 = tensor.insert %sin_tz into %Rz_2[%c0, %c1] : tensor<3x3xf64>
  %nsin_tz = arith.negf %sin_tz : f64
  %Rz_4 = tensor.insert %nsin_tz into %Rz_3[%c1, %c0] : tensor<3x3xf64>
  %Rz_5 = tensor.insert %cos_tz into %Rz_4[%c1, %c1] : tensor<3x3xf64>
  %Rz = tensor.insert %one into %Rz_5[%c2, %c2] : tensor<3x3xf64>

  %R_init = arith.constant dense<0.0> : tensor<3x3xf64>
  %R_0 = linalg.matmul ins(%Ry, %Rz : tensor<3x3xf64>, tensor<3x3xf64>) outs(%R_init : tensor<3x3xf64>) -> tensor<3x3xf64>
  %R = linalg.matmul ins(%Rx, %R_0 : tensor<3x3xf64>, tensor<3x3xf64>) outs(%R_init : tensor<3x3xf64>) -> tensor<3x3xf64>
  return %R : tensor<3x3xf64>
}

func.func @pr_body(%pose_params: tensor<3xf64>, %base_relatives: tensor<4x4xf64>) -> tensor<4x4xf64> {
  %matmul_init = arith.constant dense<0.0> : tensor<4x4xf64>
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %c3 = arith.constant 3 : index
  %tr_space = tensor.empty() : tensor<4x4xf64>
  %tr_init = linalg.fill ins(%zero: f64) outs(%tr_space: tensor<4x4xf64>) -> tensor<4x4xf64>

  %R = call @meuler_angles_to_rotation_matrix(%pose_params) : (tensor<3xf64>) -> tensor<3x3xf64>
  // %U = tensor.cast %R : tensor<3x3xf64> to tensor<*xf64>
  // call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  %tr_0 = tensor.insert_slice %R into %tr_init[0, 0] [3, 3] [1, 1] : tensor<3x3xf64> into tensor<4x4xf64>
  %tr = tensor.insert %one into %tr_0[%c3, %c3] : tensor<4x4xf64>
  %rel_slice = linalg.matmul ins(%tr, %base_relatives : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) -> tensor<4x4xf64>
  return %rel_slice : tensor<4x4xf64>
}

func.func private @printMemrefF64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func.func @main() {
  %pp = arith.constant dense<[1., 2., 3.]> : tensor<3xf64>
  %br = arith.constant dense<[
    [ 1.,  2.,  3.,  4.],
    [ 5.,  6.,  7.,  8.],
    [ 9., 10., 11., 12.],
    [13., 14., 15., 16.]
  ]> : tensor<4x4xf64>
  %f = constant @pr_body : (tensor<3xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
  %df = standalone.grad %f {of = [0]} : (tensor<3xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>, (tensor<3xf64>, tensor<4x4xf64>) -> tensor<3xf64>
  %res = call_indirect %df(%pp, %br) : (tensor<3xf64>, tensor<4x4xf64>) -> tensor<3xf64>
  %U = tensor.cast %res : tensor<3xf64> to tensor<*xf64>
  // %res = call @pr_body(%pp, %br) : (tensor<3xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
  // %U = tensor.cast %res : tensor<4x4xf64> to tensor<*xf64>
  call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  return
}
