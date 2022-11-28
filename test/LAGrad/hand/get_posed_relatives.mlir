func private @meuler_angles_to_rotation_matrix(%xyz: tensor<3xf64>) -> tensor<3x3xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %tx = tensor.extract %xyz[%c0] : tensor<3xf64>
  %ty = tensor.extract %xyz[%c2] : tensor<3xf64>
  %tz = tensor.extract %xyz[%c1] : tensor<3xf64>

  %Rx_0 = linalg.init_tensor [3, 3] : tensor<3x3xf64>
  %Rx_1 = linalg.fill(%zero, %Rx_0) : f64, tensor<3x3xf64> -> tensor<3x3xf64>
  %cos_tx = math.cos %tx : f64
  %Rx_2 = tensor.insert %cos_tx into %Rx_1[%c1, %c1] : tensor<3x3xf64>
  %sin_tx = math.sin %tx : f64
  %Rx_3 = tensor.insert %sin_tx into %Rx_2[%c1, %c2] : tensor<3x3xf64>
  %nsin_tx = arith.negf %sin_tx : f64
  %Rx_4 = tensor.insert %nsin_tx into %Rx_3[%c2, %c1] : tensor<3x3xf64>
  %Rx_5 = tensor.insert %cos_tx into %Rx_4[%c2, %c2] : tensor<3x3xf64>
  %Rx = tensor.insert %one into %Rx_5[%c0, %c0] : tensor<3x3xf64>

  %Ry_0 = linalg.init_tensor [3, 3] : tensor<3x3xf64>
  %Ry_1 = linalg.fill(%zero, %Ry_0) : f64, tensor<3x3xf64> -> tensor<3x3xf64>
  %cos_ty = math.cos %ty : f64
  %Ry_2 = tensor.insert %cos_ty into %Ry_1[%c0, %c0] : tensor<3x3xf64>
  %sin_ty = math.sin %ty : f64
  %Ry_3 = tensor.insert %sin_ty into %Ry_2[%c0, %c2] : tensor<3x3xf64>
  %nsin_ty = arith.negf %sin_ty : f64
  %Ry_4 = tensor.insert %nsin_ty into %Ry_3[%c2, %c0] : tensor<3x3xf64>
  %Ry_5 = tensor.insert %cos_ty into %Ry_4[%c2, %c2] : tensor<3x3xf64>
  %Ry = tensor.insert %one into %Ry_5[%c1, %c1] : tensor<3x3xf64>

  %Rz_0 = linalg.init_tensor [3, 3] : tensor<3x3xf64>
  %Rz_1 = linalg.fill(%zero, %Rz_0) : f64, tensor<3x3xf64> -> tensor<3x3xf64>
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

func @mget_posed_relatives(%base_relatives: tensor<22x4x4xf64>, %pose_params: tensor<25x3xf64>) -> tensor<22x4x4xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cb = arith.constant 22 : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %matmul_init = arith.constant dense<0.0> : tensor<4x4xf64>
  %zero3 = arith.constant dense<0.0> : tensor<3xf64>
  %tr_space = linalg.init_tensor [4, 4] : tensor<4x4xf64>
  %tr_init = linalg.fill(%zero, %tr_space) : f64, tensor<4x4xf64> -> tensor<4x4xf64>
  %rel_space = linalg.init_tensor [22, 4, 4] : tensor<22x4x4xf64>
  %relatives = scf.for %iv = %c0 to %cb step %c1 iter_args(%rel = %rel_space) -> tensor<22x4x4xf64> {
    %pose_idx = arith.addi %iv, %c3 : index
    %pose_slice_casted = tensor.extract_slice %pose_params[%pose_idx, 0] [1, 3] [1, 1] : tensor<25x3xf64> to tensor<3xf64>

    // Bypass the issue with the function call
    %pose_slice = arith.addf %pose_slice_casted, %zero3 : tensor<3xf64>

    %R = call @meuler_angles_to_rotation_matrix(%pose_slice) : (tensor<3xf64>) -> tensor<3x3xf64>
    %tr_0 = tensor.insert_slice %R into %tr_init[0, 0] [3, 3] [1, 1] : tensor<3x3xf64> into tensor<4x4xf64>
    %tr = tensor.insert %one into %tr_0[%c3, %c3] : tensor<4x4xf64>
    %base_slice = tensor.extract_slice %base_relatives[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
    %rel_slice = linalg.matmul ins(%tr, %base_slice : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) -> tensor<4x4xf64>

    %rel_next = tensor.insert_slice %rel_slice into %rel[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4xf64> into tensor<22x4x4xf64>
    scf.yield %rel_next : tensor<22x4x4xf64>
  }
  return %relatives : tensor<22x4x4xf64>
}

func @lagrad_get_posed_relatives(%base_relatives: tensor<22x4x4xf64>, %pose_params: tensor<25x3xf64>) -> tensor<25x3xf64> {
  %f = constant @mget_posed_relatives : (tensor<22x4x4xf64>, tensor<25x3xf64>) -> tensor<22x4x4xf64>
  %df = standalone.grad %f {of = [1]} : (tensor<22x4x4xf64>, tensor<25x3xf64>) -> tensor<22x4x4xf64>, (tensor<22x4x4xf64>, tensor<25x3xf64>) -> tensor<25x3xf64>
  %res = call_indirect %df(%base_relatives, %pose_params) : (tensor<22x4x4xf64>, tensor<25x3xf64>) -> tensor<25x3xf64>
  return %res : tensor<25x3xf64>
}
