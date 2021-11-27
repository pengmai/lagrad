// The gradient of this whole function is a bunch of 1s
func @mto_pose_params(%theta: tensor<{{nbones}}xf64>) -> tensor<{{nbones + 3}}x3xf64> {
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %ones = arith.constant dense<1.0> : tensor<3xf64>
  %pose_params_space = linalg.init_tensor [{{nbones + 3}}, 3] : tensor<{{nbones + 3}}x3xf64>
  %pose_params_init = linalg.fill(%zero, %pose_params_space) : f64, tensor<{{nbones + 3}}x3xf64> -> tensor<{{nbones + 3}}x3xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %pp_1 = scf.for %iv = %c0 to %c3 step %c1 iter_args(%ppiv = %pose_params_init) -> (tensor<{{nbones + 3}}x3xf64>) {
    %th0 = tensor.extract %theta[%iv] : tensor<{{nbones}}xf64>
    %th3_idx = arith.addi %iv, %c3 : index
    %th1 = tensor.extract %theta[%th3_idx] : tensor<{{nbones}}xf64>
    %ppiv_0 = tensor.insert %th0 into %ppiv[%c0, %iv] : tensor<{{nbones + 3}}x3xf64>
    %ppiv_1 = tensor.insert %one into %ppiv_0[%c1, %iv] : tensor<{{nbones + 3}}x3xf64>
    %ppiv_2 = tensor.insert %th1 into %ppiv_1[%c2, %iv] : tensor<{{nbones + 3}}x3xf64>
    scf.yield %ppiv_2 : tensor<{{nbones + 3}}x3xf64>
  }

  %i_theta_start = arith.constant 6 : index
  %i_pose = arith.constant 5 : index
  %pp_2:3 = scf.for %iv = %c0 to %c5 step %c1 iter_args(%ppiv = %pp_1, %i_theta = %i_theta_start, %i_pp = %i_pose) -> (tensor<{{nbones + 3}}x3xf64>, index, index) {
    %th0 = tensor.extract %theta[%i_theta] : tensor<{{nbones}}xf64>
    %it_1 = arith.addi %i_theta, %c1 : index
    %th1 = tensor.extract %theta[%it_1] : tensor<{{nbones}}xf64>
    %it_2 = arith.addi %it_1, %c1 : index
    %th2 = tensor.extract %theta[%it_2] : tensor<{{nbones}}xf64>
    %it_3 = arith.addi %it_2, %c1 : index
    %th3 = tensor.extract %theta[%it_3] : tensor<{{nbones}}xf64>
    %it_4 = arith.addi %it_3, %c1 : index

    %ppiv_0 = tensor.insert %th0 into %ppiv[%i_pp, %c0] : tensor<{{nbones + 3}}x3xf64>
    %ppiv_1 = tensor.insert %th1 into %ppiv_0[%i_pp, %c1] : tensor<{{nbones + 3}}x3xf64>
    %ip_1 = arith.addi %i_pp, %c1 : index
    %ppiv_2 = tensor.insert %th2 into %ppiv_1[%ip_1, %c0] : tensor<{{nbones + 3}}x3xf64>
    %ip_2 = arith.addi %ip_1, %c1 : index
    %ppiv_3 = tensor.insert %th3 into %ppiv_2[%ip_2, %c0] : tensor<{{nbones + 3}}x3xf64>

    %ip_3 = arith.addi %ip_2, %c2 : index
    scf.yield %ppiv_3, %it_4, %ip_3 : tensor<{{nbones + 3}}x3xf64>, index, index
  }

  return %pp_2#0 : tensor<{{nbones + 3}}x3xf64>
}

func @dtopose_params(%theta: tensor<{{nbones}}xf64>) -> tensor<{{nbones}}xf64> {
  %f = constant @mto_pose_params : (tensor<{{nbones}}xf64>) -> tensor<{{nbones + 3}}x3xf64>
  %df = standalone.grad %f : (tensor<{{nbones}}xf64>) -> tensor<{{nbones + 3}}x3xf64>, (tensor<{{nbones}}xf64>) -> tensor<{{nbones}}xf64>
  %res = call_indirect %df(%theta) : (tensor<{{nbones}}xf64>) -> tensor<{{nbones}}xf64>
  return %res : tensor<{{nbones}}xf64>
}

func @euler_angles_to_rotation_matrix(%xyz: tensor<3xf64>) -> tensor<3x3xf64> {
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
  %Rx_3 = tensor.insert %sin_tx into %Rx_2[%c2, %c1] : tensor<3x3xf64>
  %nsin_tx = arith.negf %sin_tx : f64
  %Rx_4 = tensor.insert %nsin_tx into %Rx_3[%c1, %c2] : tensor<3x3xf64>
  %Rx_5 = tensor.insert %cos_tx into %Rx_4[%c2, %c2] : tensor<3x3xf64>
  %Rx = tensor.insert %one into %Rx_5[%c0, %c0] : tensor<3x3xf64>

  %Ry_0 = linalg.init_tensor [3, 3] : tensor<3x3xf64>
  %Ry_1 = linalg.fill(%zero, %Ry_0) : f64, tensor<3x3xf64> -> tensor<3x3xf64>
  %cos_ty = math.cos %ty : f64
  %Ry_2 = tensor.insert %cos_ty into %Ry_1[%c0, %c0] : tensor<3x3xf64>
  %sin_ty = math.sin %ty : f64
  %Ry_3 = tensor.insert %sin_ty into %Ry_2[%c2, %c0] : tensor<3x3xf64>
  %nsin_ty = arith.negf %sin_ty : f64
  %Ry_4 = tensor.insert %nsin_ty into %Ry_3[%c0, %c2] : tensor<3x3xf64>
  %Ry_5 = tensor.insert %cos_ty into %Ry_4[%c2, %c2] : tensor<3x3xf64>
  %Ry = tensor.insert %one into %Ry_5[%c1, %c1] : tensor<3x3xf64>

  %Rz_0 = linalg.init_tensor [3, 3] : tensor<3x3xf64>
  %Rz_1 = linalg.fill(%zero, %Rz_0) : f64, tensor<3x3xf64> -> tensor<3x3xf64>
  %cos_tz = math.cos %tz : f64
  %Rz_2 = tensor.insert %cos_tz into %Rz_1[%c0, %c0] : tensor<3x3xf64>
  %sin_tz = math.sin %tz : f64
  %Rz_3 = tensor.insert %sin_tz into %Rz_2[%c1, %c0] : tensor<3x3xf64>
  %nsin_tz = arith.negf %sin_tz : f64
  %Rz_4 = tensor.insert %nsin_tz into %Rz_3[%c0, %c1] : tensor<3x3xf64>
  %Rz_5 = tensor.insert %cos_tz into %Rz_4[%c1, %c1] : tensor<3x3xf64>
  %Rz = tensor.insert %one into %Rz_5[%c2, %c2] : tensor<3x3xf64>

  %R_init = arith.constant dense<0.0> : tensor<3x3xf64>
  %R_0 = linalg.matmul ins(%Rz, %Ry : tensor<3x3xf64>, tensor<3x3xf64>) outs(%R_init : tensor<3x3xf64>) -> tensor<3x3xf64>
  %R = linalg.matmul ins(%R_0, %Rx : tensor<3x3xf64>, tensor<3x3xf64>) outs(%R_init : tensor<3x3xf64>) -> tensor<3x3xf64>
  return %R : tensor<3x3xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %arg = arith.constant dense<[1., 2., 3.]> : tensor<3xf64>
  %f = constant @euler_angles_to_rotation_matrix : (tensor<3xf64>) -> tensor<3x3xf64>
  %df = standalone.grad %f : (tensor<3xf64>) -> tensor<3x3xf64>, (tensor<3xf64>) -> tensor<3xf64>
  %res = call_indirect %df(%arg) : (tensor<3xf64>) -> tensor<3xf64>
  %U = tensor.cast %res : tensor<3xf64> to tensor<*xf64>
  // %res = call @euler_angles_to_rotation_matrix(%arg) : (tensor<3xf64>) -> tensor<3x3xf64>
  // %U = tensor.cast %res : tensor<3x3xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}

func @mget_posed_relatives(%base_relatives: tensor<{{nbones}}x4x4xf64>, %pose_params: tensor<{{nbones + 3}}xf64>) -> tensor<{{nbones}}x4x4xf64> {
  return %base_relatives : tensor<{{nbones}}x4x4xf64>
}

// func @mget_skinned_vertex_positions(
//   %base_relatives: tensor<{{nbones}}x4x4xf64>,
//   %parents: tensor<i64>,
//   %inverse_base_absolutes: tensor<{{nbones}}x4x4xf64>,
//   %base_positions: tensor<4x{{nverts}}xf64>,
//   %weights: tensor<{{nbones}}x{{nverts}}xf64>,
//   %pose_params: tensor<{{nbones + 3}}x3xf64>
// ) -> tensor<f64> {
//   return %parents : tensor<f64>
// }
