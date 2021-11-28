// The gradient of this whole function is a bunch of 1s
func @mto_pose_params(%theta: tensor<{{ntheta}}xf64>) -> tensor<{{nbones + 3}}x3xf64> {
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
    %th0 = tensor.extract %theta[%iv] : tensor<{{ntheta}}xf64>
    %th3_idx = arith.addi %iv, %c3 : index
    %th1 = tensor.extract %theta[%th3_idx] : tensor<{{ntheta}}xf64>
    %ppiv_0 = tensor.insert %th0 into %ppiv[%c0, %iv] : tensor<{{nbones + 3}}x3xf64>
    %ppiv_1 = tensor.insert %one into %ppiv_0[%c1, %iv] : tensor<{{nbones + 3}}x3xf64>
    %ppiv_2 = tensor.insert %th1 into %ppiv_1[%c2, %iv] : tensor<{{nbones + 3}}x3xf64>
    scf.yield %ppiv_2 : tensor<{{nbones + 3}}x3xf64>
  }

  %i_theta_start = arith.constant 6 : index
  %i_pose = arith.constant 5 : index
  %pp_2:3 = scf.for %iv = %c0 to %c5 step %c1 iter_args(%ppiv = %pp_1, %i_theta = %i_theta_start, %i_pp = %i_pose) -> (tensor<{{nbones + 3}}x3xf64>, index, index) {
    %th0 = tensor.extract %theta[%i_theta] : tensor<{{ntheta}}xf64>
    %it_1 = arith.addi %i_theta, %c1 : index
    %th1 = tensor.extract %theta[%it_1] : tensor<{{ntheta}}xf64>
    %it_2 = arith.addi %it_1, %c1 : index
    %th2 = tensor.extract %theta[%it_2] : tensor<{{ntheta}}xf64>
    %it_3 = arith.addi %it_2, %c1 : index
    %th3 = tensor.extract %theta[%it_3] : tensor<{{ntheta}}xf64>
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

func @dtopose_params(%theta: tensor<{{ntheta}}xf64>) -> tensor<{{ntheta}}xf64> {
  %f = constant @mto_pose_params : (tensor<{{ntheta}}xf64>) -> tensor<{{nbones + 3}}x3xf64>
  %df = standalone.grad %f : (tensor<{{ntheta}}xf64>) -> tensor<{{nbones + 3}}x3xf64>, (tensor<{{ntheta}}xf64>) -> tensor<{{ntheta}}xf64>
  %res = call_indirect %df(%theta) : (tensor<{{ntheta}}xf64>) -> tensor<{{ntheta}}xf64>
  return %res : tensor<{{ntheta}}xf64>
}

// This produces transposed results relative to ADBench.
func @meuler_angles_to_rotation_matrix(%xyz: tensor<3xf64>) -> tensor<3x3xf64> attributes {pure = true} {
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

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @test(%theta : tensor<{{ntheta}}xf64>, %base_relatives: tensor<{{nbones}}x4x4xf64>) -> tensor<{{nbones}}x4x4xf64> {
  %pose_params = call @mto_pose_params(%theta) : (tensor<{{ntheta}}xf64>) -> tensor<{{nbones + 3}}x3xf64>
  %res = call @mget_posed_relatives(%base_relatives, %pose_params) : (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones}}x4x4xf64>
  return %res : tensor<{{nbones}}x4x4xf64>
}

func @dtest(%theta : tensor<{{ntheta}}xf64>, %base_relatives: tensor<{{nbones}}x4x4xf64>) -> tensor<{{ntheta}}xf64> {
  %f = constant @test : (tensor<{{ntheta}}xf64>, tensor<{{nbones}}x4x4xf64>) -> tensor<{{nbones}}x4x4xf64>
  %df = standalone.grad %f : (tensor<{{ntheta}}xf64>, tensor<{{nbones}}x4x4xf64>) -> tensor<{{nbones}}x4x4xf64>, (tensor<{{ntheta}}xf64>, tensor<{{nbones}}x4x4xf64>) -> tensor<{{ntheta}}xf64>
  %res = call_indirect %df(%theta, %base_relatives) : (tensor<{{ntheta}}xf64>, tensor<{{nbones}}x4x4xf64>) -> tensor<{{ntheta}}xf64>
  return %res : tensor<{{ntheta}}xf64>
}

func @mget_posed_relatives(%base_relatives: tensor<{{nbones}}x4x4xf64>, %pose_params: tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones}}x4x4xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cb = arith.constant {{nbones}} : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %matmul_init = arith.constant dense<0.0> : tensor<4x4xf64>
  %tr_space = linalg.init_tensor [4, 4] : tensor<4x4xf64>
  %tr_init = linalg.fill(%zero, %tr_space) : f64, tensor<4x4xf64> -> tensor<4x4xf64>
  %rel_space = linalg.init_tensor [{{nbones}}, 4, 4] : tensor<{{nbones}}x4x4xf64>
  %relatives = scf.for %iv = %c0 to %cb step %c1 iter_args(%rel = %rel_space) -> tensor<{{nbones}}x4x4xf64> {
    %pose_idx = arith.addi %iv, %c3 : index
    %pose_slice_casted = tensor.extract_slice %pose_params[%pose_idx, 0] [1, 3] [1, 1] : tensor<{{nbones + 3}}x3xf64> to tensor<3xf64>

    // Bypass the issue with the function call
    %pose_slice = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%pose_slice_casted : tensor<3xf64>) outs(%pose_slice_casted : tensor<3xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = arith.addf %arg0, %zero : f64
      linalg.yield %0 : f64
    } -> tensor<3xf64>

    %R = call @meuler_angles_to_rotation_matrix(%pose_slice) : (tensor<3xf64>) -> tensor<3x3xf64>
    %tr_0 = tensor.insert_slice %R into %tr_init[0, 0] [3, 3] [1, 1] : tensor<3x3xf64> into tensor<4x4xf64>
    %tr = tensor.insert %one into %tr_0[%c3, %c3] : tensor<4x4xf64>
    %base_slice = tensor.extract_slice %base_relatives[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<{{nbones}}x4x4xf64> to tensor<4x4xf64>
    %rel_slice = linalg.matmul ins(%tr, %base_slice : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) -> tensor<4x4xf64>

    %rel_next = tensor.insert_slice %rel_slice into %rel[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4xf64> into tensor<{{nbones}}x4x4xf64>
    scf.yield %rel_next : tensor<{{nbones}}x4x4xf64>
  }
  return %relatives : tensor<{{nbones}}x4x4xf64>
}

func @dget_posed_relatives(%base_relatives: tensor<{{nbones}}x4x4xf64>, %pose_params: tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones + 3}}x3xf64> {
  %f = constant @mget_posed_relatives : (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones}}x4x4xf64>
  %df = standalone.grad %f {of = [1]} : (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones}}x4x4xf64>, (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones + 3}}x3xf64>
  %res = call_indirect %df(%base_relatives, %pose_params) : (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones + 3}}x3xf64>
  return %res : tensor<{{nbones + 3}}x3xf64>
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
