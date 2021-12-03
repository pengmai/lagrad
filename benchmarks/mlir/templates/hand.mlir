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

func @axis_angles_to_rotation_matrix(%angle_axis: tensor<3xf64>) -> tensor<3x3xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %eye = arith.constant sparse<[[0, 0], [1, 1], [2, 2]], 1.0> : tensor<3x3xf64>
  %sqsum_init = arith.constant dense<0.0> : tensor<f64>
  %sqsum = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]}
    ins(%angle_axis : tensor<3xf64>)
    outs(%sqsum_init : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = arith.mulf %arg0, %arg0 : f64
    %1 = arith.addf %0, %arg1 : f64
    linalg.yield %1 : f64
  } -> tensor<f64>
  %sqsum_v = tensor.extract %sqsum[] : tensor<f64>
  %norm = math.sqrt %sqsum_v : f64
  %tol = arith.constant 1.0e-4 : f64
  %pred = arith.cmpf "olt", %norm, %tol : f64
  %res = scf.if %pred -> tensor<3x3xf64> {
    scf.yield %eye : tensor<3x3xf64>
  } else {
    %res_space = linalg.init_tensor [3, 3] : tensor<3x3xf64>
    %x_0 = tensor.extract %angle_axis[%c0] : tensor<3xf64>
    %y_0 = tensor.extract %angle_axis[%c1] : tensor<3xf64>
    %z_0 = tensor.extract %angle_axis[%c2] : tensor<3xf64>
    %x = arith.divf %x_0, %norm : f64
    %y = arith.divf %y_0, %norm : f64
    %z = arith.divf %z_0, %norm : f64
    %s = math.sin %norm : f64
    %c = math.cos %norm : f64

    %xx = arith.mulf %x, %x : f64
    %xy = arith.mulf %x, %y : f64
    %xz = arith.mulf %x, %z : f64
    %onemcos = arith.subf %one, %c : f64
    %zs = arith.mulf %z, %s : f64
    %ys = arith.mulf %y, %s : f64
    // First row
    
    %onemxx = arith.subf %one, %xx : f64
    %res_00_0 = arith.mulf %onemxx, %c : f64
    %res_00_1 = arith.addf %xx, %res_00_0 : f64
    %res_0 = tensor.insert %res_00_1 into %res_space[%c0, %c0] : tensor<3x3xf64>
    scf.yield %res_0 : tensor<3x3xf64>
  }
  return %res : tensor<3x3xf64>
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

func @test(%theta : tensor<{{ntheta}}xf64>, %base_relatives: tensor<{{nbones}}x4x4xf64>, %parents: tensor<{{nbones}}xi32>) -> tensor<{{nbones}}x4x4xf64> {
  %pose_params = call @mto_pose_params(%theta) : (tensor<{{ntheta}}xf64>) -> tensor<{{nbones + 3}}x3xf64>
  %res = call @mget_posed_relatives(%base_relatives, %pose_params) : (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones}}x4x4xf64>
  %abs = call @mrelatives_to_absolutes(%res, %parents) : (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones}}xi32>) -> tensor<{{nbones}}x4x4xf64>
  return %abs : tensor<{{nbones}}x4x4xf64>
}

func @dtest(%theta : tensor<{{ntheta}}xf64>, %base_relatives: tensor<{{nbones}}x4x4xf64>, %parents: tensor<{{nbones}}xi32>) -> tensor<{{ntheta}}xf64> {
  %f = constant @test : (tensor<{{ntheta}}xf64>, tensor<{{nbones}}x4x4xf64>, tensor<{{nbones}}xi32>) -> tensor<{{nbones}}x4x4xf64>
  %df = standalone.grad %f {of = [0]} : (tensor<{{ntheta}}xf64>, tensor<{{nbones}}x4x4xf64>, tensor<{{nbones}}xi32>) -> tensor<{{nbones}}x4x4xf64>, (tensor<{{ntheta}}xf64>, tensor<{{nbones}}x4x4xf64>, tensor<{{nbones}}xi32>) -> tensor<{{ntheta}}xf64>
  %res = call_indirect %df(%theta, %base_relatives, %parents) : (tensor<{{ntheta}}xf64>, tensor<{{nbones}}x4x4xf64>, tensor<{{nbones}}xi32>) -> tensor<{{ntheta}}xf64>
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

func @mrelatives_to_absolutes(%relatives: tensor<{{nbones}}x4x4xf64>, %parents: tensor<{{nbones}}xi32>) -> tensor<{{nbones}}x4x4xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n_one = arith.constant -1 : i32
  %cb = arith.constant {{nbones}} : index
  %matmul_init = arith.constant dense<0.0> : tensor<4x4xf64>
  %absolute_space = linalg.init_tensor [{{nbones}}, 4, 4] : tensor<{{nbones}}x4x4xf64>
  %absolutes = scf.for %iv = %c0 to %cb step %c1 iter_args(%a_iter = %absolute_space) -> (tensor<{{nbones}}x4x4xf64>) {
    %parent_i = tensor.extract %parents[%iv] : tensor<{{nbones}}xi32>
    %pred = arith.cmpi "eq", %parent_i, %n_one : i32
    %rel_i = tensor.extract_slice %relatives[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<{{nbones}}x4x4xf64> to tensor<4x4xf64>
    %result = scf.if %pred -> tensor<4x4xf64> {
      scf.yield %rel_i : tensor<4x4xf64>
    } else {
      %parent_idx = arith.index_cast %parent_i : i32 to index
      %abs_p = tensor.extract_slice %a_iter[%parent_idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<{{nbones}}x4x4xf64> to tensor<4x4xf64>
      %abs_i = linalg.matmul ins(%rel_i, %abs_p : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) -> tensor<4x4xf64>
      scf.yield %abs_i : tensor<4x4xf64>
    }
    %a_next = tensor.insert_slice %result into %a_iter[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4xf64> into tensor<{{nbones}}x4x4xf64>
    scf.yield %a_next : tensor<{{nbones}}x4x4xf64>
  }
  return %absolutes : tensor<{{nbones}}x4x4xf64>
}

func @dget_posed_relatives(%base_relatives: tensor<{{nbones}}x4x4xf64>, %pose_params: tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones + 3}}x3xf64> {
  %f = constant @mget_posed_relatives : (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones}}x4x4xf64>
  %df = standalone.grad %f {of = [1]} : (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones}}x4x4xf64>, (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones + 3}}x3xf64>
  %res = call_indirect %df(%base_relatives, %pose_params) : (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones + 3}}x3xf64>
  return %res : tensor<{{nbones + 3}}x3xf64>
}

func @mget_skinned_vertex_positions(
  %base_relatives: tensor<{{nbones}}x4x4xf64>,
  %parents: tensor<{{nbones}}xi32>,
  %inverse_base_absolutes: tensor<{{nbones}}x4x4xf64>,
  %base_positions: tensor<{{nverts}}x4xf64>,
  %weights: tensor<{{nbones}}x{{nverts}}xf64>,
  %pose_params: tensor<{{nbones + 3}}x3xf64>
) -> tensor<{{nverts}}x3xf64> {
  %relatives = call @mget_posed_relatives(%base_relatives, %pose_params) : (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones + 3}}x3xf64>) -> tensor<{{nbones}}x4x4xf64>
  %absolutes = call @mrelatives_to_absolutes(%relatives, %parents) : (tensor<{{nbones}}x4x4xf64>, tensor<{{nbones}}xi32>) -> tensor<{{nbones}}x4x4xf64>
  %transforms_init = arith.constant dense<0.0> : tensor<{{nbones}}x4x4xf64>
  %transforms = linalg.batch_matmul ins(%inverse_base_absolutes, %absolutes : tensor<{{nbones}}x4x4xf64>, tensor<{{nbones}}x4x4xf64>) outs(%transforms_init: tensor<{{nbones}}x4x4xf64>) -> tensor<{{nbones}}x4x4xf64>

  %positions_init = arith.constant dense<0.0> : tensor<{{nverts}}x3xf64>
  %zero = arith.constant 0.0 : f64
  %curr_positions_space = linalg.init_tensor [{{nverts}}, 4] : tensor<{{nverts}}x4xf64>
  %curr_positions_init = linalg.fill(%zero, %curr_positions_space) : f64, tensor<{{nverts}}x4xf64> -> tensor<{{nverts}}x4xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cb = arith.constant {{nbones}} : index
  %positions = scf.for %iv = %c0 to %cb step %c1 iter_args(%positions_i = %positions_init) -> tensor<{{nverts}}x3xf64> {
    %transforms_slice = tensor.extract_slice %transforms[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<{{nbones}}x4x4xf64> to tensor<4x4xf64>
    %curr_positions = linalg.generic
      {
        doc = "Column-major matrix multiplication",
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d2, d0)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d0)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      }
      ins(%transforms_slice, %base_positions : tensor<4x4xf64>, tensor<{{nverts}}x4xf64>)
      outs(%curr_positions_init : tensor<{{nverts}}x4xf64>) {
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      %0 = arith.mulf %arg0, %arg1 : f64
      %1 = arith.addf %0, %arg2 : f64
      linalg.yield %1 : f64
    } -> tensor<{{nverts}}x4xf64>
    // %curr_positions = linalg.matmul ins(%base_positions, %transforms_slice : tensor<{{nverts}}x4xf64>, tensor<4x4xf64>) outs(%curr_positions_init : tensor<{{nverts}}x4xf64>) -> tensor<{{nverts}}x4xf64>
    %cp_slice = tensor.extract_slice %curr_positions[0, 0] [{{nverts}}, 3] [1, 1] : tensor<{{nverts}}x4xf64> to tensor<{{nverts}}x3xf64>
    %weight_slice = tensor.extract_slice %weights[%iv, 0] [1, {{nverts}}] [1, 1] : tensor<{{nbones}}x{{nverts}}xf64> to tensor<{{nverts}}xf64>

    // %p = arith.cmpi "eq", %c0, %iv : index
    // scf.if %p {
    //   %U = tensor.cast %weight_slice : tensor<{{nverts}}xf64> to tensor<*xf64>
    //   call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
    // }

    %positions_next = linalg.generic
      {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%cp_slice, %weight_slice : tensor<{{nverts}}x3xf64>, tensor<{{nverts}}xf64>)
      outs(%positions_i : tensor<{{nverts}}x3xf64>) {
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      %0 = arith.mulf %arg0, %arg1 : f64
      %1 = arith.addf %0, %arg2 : f64
      linalg.yield %1 : f64
    } -> tensor<{{nverts}}x3xf64>
    scf.yield %positions_next : tensor<{{nverts}}x3xf64>
  }

  // %p_slice = tensor.extract_slice %positions[0, 0] [1, 3] [1, 1] : tensor<3x{{nverts}}xf64> to tensor<3xf64>
  // %U = tensor.cast %p_slice : tensor<3xf64> to tensor<*xf64>
  // call @print_memref_f64(%U) : (tensor<*xf64>) -> ()

  return %positions : tensor<{{nverts}}x3xf64>
}

func @mlir_hand_objective(
  %theta : tensor<{{ntheta}}xf64>,
  %parents: tensor<{{nbones}}xi32>,
  %base_relatives: tensor<{{nbones}}x4x4xf64>,
  %inverse_base_absolutes: tensor<{{nbones}}x4x4xf64>,
  %base_positions: tensor<{{nverts}}x4xf64>,
  %weights: tensor<{{nbones}}x{{nverts}}xf64>
) {
  %pose_params = call @mto_pose_params(%theta) : (tensor<{{ntheta}}xf64>) -> tensor<{{nbones + 3}}x3xf64>
  %res = call @mget_skinned_vertex_positions(
    %base_relatives,
    %parents,
    %inverse_base_absolutes,
    %base_positions,
    %weights,
    %pose_params
  ) : (
    tensor<{{nbones}}x4x4xf64>,
    tensor<{{nbones}}xi32>,
    tensor<{{nbones}}x4x4xf64>,
    tensor<{{nverts}}x4xf64>,
    tensor<{{nbones}}x{{nverts}}xf64>,
    tensor<{{nbones + 3}}x3xf64>
  ) -> tensor<{{nverts}}x3xf64>
  return
}
