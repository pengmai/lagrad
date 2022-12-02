func.func private @mto_pose_params(%theta: tensor<26xf64>) -> tensor<25x3xf64> {
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %ones = arith.constant dense<1.0> : tensor<3xf64>
  %pose_params_space = tensor.empty() : tensor<25x3xf64>
  %pose_params_init = linalg.fill ins(%zero: f64) outs(%pose_params_space: tensor<25x3xf64>) -> tensor<25x3xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %pp_1 = scf.for %iv = %c0 to %c3 step %c1 iter_args(%ppiv = %pose_params_init) -> (tensor<25x3xf64>) {
    %th0 = tensor.extract %theta[%iv] : tensor<26xf64>
    %th3_idx = arith.addi %iv, %c3 : index
    %th1 = tensor.extract %theta[%th3_idx] : tensor<26xf64>
    %ppiv_0 = tensor.insert %th0 into %ppiv[%c0, %iv] : tensor<25x3xf64>
    %ppiv_1 = tensor.insert %one into %ppiv_0[%c1, %iv] : tensor<25x3xf64>
    %ppiv_2 = tensor.insert %th1 into %ppiv_1[%c2, %iv] : tensor<25x3xf64>
    scf.yield %ppiv_2 : tensor<25x3xf64>
  }

  %i_theta_start = arith.constant 6 : index
  %i_pose = arith.constant 5 : index
  %pp_2:3 = scf.for %iv = %c0 to %c5 step %c1 iter_args(%ppiv = %pp_1, %i_theta = %i_theta_start, %i_pp = %i_pose) -> (tensor<25x3xf64>, index, index) {
    %th0 = tensor.extract %theta[%i_theta] : tensor<26xf64>
    %it_1 = arith.addi %i_theta, %c1 : index
    %th1 = tensor.extract %theta[%it_1] : tensor<26xf64>
    %it_2 = arith.addi %it_1, %c1 : index
    %th2 = tensor.extract %theta[%it_2] : tensor<26xf64>
    %it_3 = arith.addi %it_2, %c1 : index
    %th3 = tensor.extract %theta[%it_3] : tensor<26xf64>
    %it_4 = arith.addi %it_3, %c1 : index

    %ppiv_0 = tensor.insert %th0 into %ppiv[%i_pp, %c0] : tensor<25x3xf64>
    %ppiv_1 = tensor.insert %th1 into %ppiv_0[%i_pp, %c1] : tensor<25x3xf64>
    %ip_1 = arith.addi %i_pp, %c1 : index
    %ppiv_2 = tensor.insert %th2 into %ppiv_1[%ip_1, %c0] : tensor<25x3xf64>
    %ip_2 = arith.addi %ip_1, %c1 : index
    %ppiv_3 = tensor.insert %th3 into %ppiv_2[%ip_2, %c0] : tensor<25x3xf64>

    %ip_3 = arith.addi %ip_2, %c2 : index
    scf.yield %ppiv_3, %it_4, %ip_3 : tensor<25x3xf64>, index, index
  }

  return %pp_2#0 : tensor<25x3xf64>
}

func.func private @mangle_axis_to_rotation_matrix(%angle_axis: tensor<3xf64>) -> tensor<3x3xf64> {
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
  %res_space = tensor.empty() : tensor<3x3xf64>
  %tol = arith.constant 1.0e-4 : f64
  %pred = arith.cmpf "olt", %norm, %tol : f64
  %res = scf.if %pred -> tensor<3x3xf64> {
    scf.yield %eye : tensor<3x3xf64>
  } else {
    %x_0 = tensor.extract %angle_axis[%c0] : tensor<3xf64>
    %y_0 = tensor.extract %angle_axis[%c1] : tensor<3xf64>
    %z_0 = tensor.extract %angle_axis[%c2] : tensor<3xf64>
    %x = arith.divf %x_0, %norm : f64
    %y = arith.divf %y_0, %norm : f64
    %z = arith.divf %z_0, %norm : f64
    %s = math.sin %norm : f64
    %c = math.cos %norm : f64

    %xx = arith.mulf %x, %x : f64
    %yy = arith.mulf %y, %y : f64
    %zz = arith.mulf %z, %z : f64
    %xy = arith.mulf %x, %y : f64
    %xz = arith.mulf %x, %z : f64
    %yz = arith.mulf %y, %z : f64
    %onemcos = arith.subf %one, %c : f64
    %xs = arith.mulf %x, %s : f64
    %ys = arith.mulf %y, %s : f64
    %zs = arith.mulf %z, %s : f64
    // First row
    %onemxx = arith.subf %one, %xx : f64
    %res_00_0 = arith.mulf %onemxx, %c : f64
    %res_00_1 = arith.addf %xx, %res_00_0 : f64
    %res_0 = tensor.insert %res_00_1 into %res_space[%c0, %c0] : tensor<3x3xf64>
    %res_10_0 = arith.mulf %xy, %onemcos : f64
    %res_10 = arith.subf %res_10_0, %zs : f64
    %res_1 = tensor.insert %res_10 into %res_0[%c1, %c0] : tensor<3x3xf64>
    %res_20_0 = arith.mulf %xz, %onemcos : f64
    %res_20 = arith.addf %res_20_0, %ys : f64
    %res_2 = tensor.insert %res_20 into %res_1[%c2, %c0] : tensor<3x3xf64>

    // Second row
    %res_01_0 = arith.mulf %xy, %onemcos : f64
    %res_01 = arith.addf %res_01_0, %zs : f64
    %res_3 = tensor.insert %res_01 into %res_2[%c0, %c1] : tensor<3x3xf64>
    %onemyy = arith.subf %one, %yy : f64
    %res_11_0 = arith.mulf %onemyy, %c : f64
    %res_11 = arith.addf %yy, %res_11_0 : f64
    %res_4 = tensor.insert %res_11 into %res_3[%c1, %c1] : tensor<3x3xf64>
    %res_21_0 = arith.mulf %yz, %onemcos : f64
    %res_21 = arith.subf %res_21_0, %xs : f64
    %res_5 = tensor.insert %res_21 into %res_4[%c2, %c1] : tensor<3x3xf64>

    // Third row
    %res_02_0 = arith.mulf %xz, %onemcos : f64
    %res_02 = arith.subf %res_02_0, %ys : f64
    %res_6 = tensor.insert %res_02 into %res_5[%c0, %c2] : tensor<3x3xf64>
    %res_12_0 = arith.mulf %yz, %onemcos : f64
    %res_12 = arith.addf %res_12_0, %xs : f64
    %res_7 = tensor.insert %res_12 into %res_6[%c1, %c2] : tensor<3x3xf64>
    %onemzz = arith.subf %one, %zz : f64
    %res_22_0 = arith.mulf %onemzz, %c : f64
    %res_22 = arith.addf %zz, %res_22_0 : f64
    %res_8 = tensor.insert %res_22 into %res_7[%c2, %c2] : tensor<3x3xf64>
    scf.yield %res_8 : tensor<3x3xf64>
  }
  return %res : tensor<3x3xf64>
}

// This produces transposed results relative to ADBench.
func.func private @meuler_angles_to_rotation_matrix(%xyz: tensor<3xf64>) -> tensor<3x3xf64> attributes {pure = true} {
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

// func.func private @printMemrefF64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func.func private @mget_posed_relatives(%base_relatives: tensor<22x4x4xf64>, %pose_params: tensor<25x3xf64>) -> tensor<22x4x4xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cb = arith.constant 22 : index
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %matmul_init = arith.constant dense<0.0> : tensor<4x4xf64>
  %tr_space = tensor.empty() : tensor<4x4xf64>
  %zero3 = arith.constant dense<0.0> : tensor<3xf64>
  %tr_init = linalg.fill ins(%zero: f64) outs(%tr_space: tensor<4x4xf64>) -> tensor<4x4xf64>
  %rel_space = tensor.empty() : tensor<22x4x4xf64>
  %relatives = scf.for %iv = %c0 to %cb step %c1 iter_args(%rel = %rel_space) -> tensor<22x4x4xf64> {
    %pose_idx = arith.addi %iv, %c3 : index
    %pose_slice_casted = tensor.extract_slice %pose_params[%pose_idx, 0] [1, 3] [1, 1] : tensor<25x3xf64> to tensor<3xf64>

    // Bypass the issue with the function call
    %pose_slice = arith.addf %pose_slice_casted, %zero3 : tensor<3xf64>

    %R = func.call @meuler_angles_to_rotation_matrix(%pose_slice) : (tensor<3xf64>) -> tensor<3x3xf64>
    %tr_0 = tensor.insert_slice %R into %tr_init[0, 0] [3, 3] [1, 1] : tensor<3x3xf64> into tensor<4x4xf64>
    %tr = tensor.insert %one into %tr_0[%c3, %c3] : tensor<4x4xf64>
    %base_slice = tensor.extract_slice %base_relatives[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
    %rel_slice = linalg.matmul ins(%tr, %base_slice : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) -> tensor<4x4xf64>

    %rel_next = tensor.insert_slice %rel_slice into %rel[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4xf64> into tensor<22x4x4xf64>
    scf.yield %rel_next : tensor<22x4x4xf64>
  }
  return %relatives : tensor<22x4x4xf64>
}

func.func private @mrelatives_to_absolutes(%relatives: tensor<22x4x4xf64>, %parents: tensor<22xi32>) -> tensor<22x4x4xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n_one = arith.constant -1 : i32
  %cb = arith.constant 22 : index
  %matmul_init = arith.constant dense<0.0> : tensor<4x4xf64>
  %absolute_space = tensor.empty() : tensor<22x4x4xf64>
  %absolutes = scf.for %iv = %c0 to %cb step %c1 iter_args(%a_iter = %absolute_space) -> (tensor<22x4x4xf64>) {
    %parent_i = tensor.extract %parents[%iv] : tensor<22xi32>
    %rel_i = tensor.extract_slice %relatives[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
    %pred = arith.cmpi "eq", %parent_i, %n_one : i32
    %result = scf.if %pred -> tensor<4x4xf64> {
      scf.yield %rel_i : tensor<4x4xf64>
    } else {
      %parent_idx = arith.index_cast %parent_i : i32 to index
      %abs_p = tensor.extract_slice %a_iter[%parent_idx, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
      %abs_i = linalg.matmul ins(%rel_i, %abs_p : tensor<4x4xf64>, tensor<4x4xf64>) outs(%matmul_init : tensor<4x4xf64>) -> tensor<4x4xf64>
      scf.yield %abs_i : tensor<4x4xf64>
    }
    %a_next = tensor.insert_slice %result into %a_iter[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<4x4xf64> into tensor<22x4x4xf64>
    scf.yield %a_next : tensor<22x4x4xf64>
  }
  return %absolutes : tensor<22x4x4xf64>
}

// A helper to quickly get the transforms input into the skinned vertex subset.
func.func @HELPER_get_transforms(
  %theta : tensor<26xf64>,
  %parents: tensor<22xi32>,
  %base_relatives: tensor<22x4x4xf64>,
  %inverse_base_absolutes: tensor<22x4x4xf64>
) -> tensor<22x4x4xf64> {
  %pose_params = call @mto_pose_params(%theta) : (tensor<26xf64>) -> tensor<25x3xf64>
  // Inlined get_skinned_vertex_positions
  %relatives = call @mget_posed_relatives(%base_relatives, %pose_params) : (tensor<22x4x4xf64>, tensor<25x3xf64>) -> tensor<22x4x4xf64>
  %absolutes = call @mrelatives_to_absolutes(%relatives, %parents) : (tensor<22x4x4xf64>, tensor<22xi32>) -> tensor<22x4x4xf64>
  %transforms_init = arith.constant dense<0.0> : tensor<22x4x4xf64>
  %transforms = linalg.batch_matmul ins(%inverse_base_absolutes, %absolutes : tensor<22x4x4xf64>, tensor<22x4x4xf64>) outs(%transforms_init: tensor<22x4x4xf64>) -> tensor<22x4x4xf64>
  return %transforms : tensor<22x4x4xf64>
}

func.func @mlir_hand_objective(
  %theta : tensor<26xf64>,
  %parents: tensor<22xi32>,
  %base_relatives: tensor<22x4x4xf64>,
  %inverse_base_absolutes: tensor<22x4x4xf64>,
  %base_positions: tensor<544x4xf64>,
  %weights: tensor<544x22xf64>,
  %correspondences: tensor<?xi32>,
  %points : tensor<?x3xf64>
) -> tensor<?x3xf64> {
  %pose_params = call @mto_pose_params(%theta) : (tensor<26xf64>) -> tensor<25x3xf64>
  // Inlined get_skinned_vertex_positions
  %relatives = call @mget_posed_relatives(%base_relatives, %pose_params) : (tensor<22x4x4xf64>, tensor<25x3xf64>) -> tensor<22x4x4xf64>
  %absolutes = call @mrelatives_to_absolutes(%relatives, %parents) : (tensor<22x4x4xf64>, tensor<22xi32>) -> tensor<22x4x4xf64>
  %transforms_init = arith.constant dense<0.0> : tensor<22x4x4xf64>
  %transforms = linalg.batch_matmul ins(%inverse_base_absolutes, %absolutes : tensor<22x4x4xf64>, tensor<22x4x4xf64>) outs(%transforms_init: tensor<22x4x4xf64>) -> tensor<22x4x4xf64>

  %positions_init = arith.constant dense<0.0> : tensor<544x3xf64>
  %zero = arith.constant 0.0 : f64
  %curr_positions_init = arith.constant dense<0.0> : tensor<544x4xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cb = arith.constant 22 : index
  %positions = scf.for %iv = %c0 to %cb step %c1 iter_args(%positions_i = %positions_init) -> tensor<544x3xf64> {
    %transforms_slice = tensor.extract_slice %transforms[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
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
      ins(%transforms_slice, %base_positions : tensor<4x4xf64>, tensor<544x4xf64>)
      outs(%curr_positions_init : tensor<544x4xf64>) {
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      %0 = arith.mulf %arg0, %arg1 : f64
      %1 = arith.addf %0, %arg2 : f64
      linalg.yield %1 : f64
    } -> tensor<544x4xf64>
    %cp_slice = tensor.extract_slice %curr_positions[0, 0] [544, 3] [1, 1] : tensor<544x4xf64> to tensor<544x3xf64>
    %weight_slice = tensor.extract_slice %weights[0, %iv] [544, 1] [1, 1] : tensor<544x22xf64> to tensor<544xf64>

    %positions_next = linalg.generic
      {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%cp_slice, %weight_slice : tensor<544x3xf64>, tensor<544xf64>)
      outs(%positions_i : tensor<544x3xf64>) {
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      %0 = arith.mulf %arg0, %arg1 : f64
      %1 = arith.addf %0, %arg2 : f64
      linalg.yield %1 : f64
    } -> tensor<544x3xf64>
    scf.yield %positions_next : tensor<544x3xf64>
  }
  // return %positions : tensor<544x3xf64>

  // Inlined apply_global_transform
  %angle_axis = tensor.extract_slice %pose_params[0, 0] [1, 3] [1, 1] : tensor<25x3xf64> to tensor<3xf64>
  %pose_slice = tensor.extract_slice %pose_params[1, 0] [1, 3] [1, 1] : tensor<25x3xf64> to tensor<3xf64>
  %pose_slice_2 = tensor.extract_slice %pose_params[2, 0] [1, 3] [1, 1] : tensor<25x3xf64> to tensor<3xf64>
  %R_0 = call @mangle_axis_to_rotation_matrix(%angle_axis) : (tensor<3xf64>) -> tensor<3x3xf64>

  %R_space = arith.constant dense<0.0> : tensor<3x3xf64>
  %R = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%R_0, %pose_slice : tensor<3x3xf64>, tensor<3xf64>)
    outs(%R_space : tensor<3x3xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  } -> tensor<3x3xf64>

  %tmp_init = arith.constant dense<0.0> : tensor<544x3xf64>
  %tmp = linalg.generic
    {
      doc = "Column-major matrix multiplication",
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2, d0)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d0)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    }
    ins(%R, %positions : tensor<3x3xf64>, tensor<544x3xf64>)
    outs(%tmp_init : tensor<544x3xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    linalg.yield %1 : f64
  } -> tensor<544x3xf64>

  %vertex_positions = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%tmp, %pose_slice_2 : tensor<544x3xf64>, tensor<3xf64>)
    outs(%tmp_init : tensor<544x3xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.addf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  } -> tensor<544x3xf64>
  // return %vertex_positions : tensor<544x3xf64>
  // End inline apply_global_transform
  // End inline get_skinned_vertex_positions

  %cpts = tensor.dim %points, %c0 : tensor<?x3xf64>
  %c3 = arith.constant 3 : index
  %err_init = tensor.empty(%cpts) : tensor<?x3xf64>
  %err = scf.for %iv = %c0 to %cpts step %c1 iter_args(%e_outer = %err_init) -> tensor<?x3xf64> {
    %err_partial = scf.for %jv = %c0 to %c3 step %c1 iter_args(%e_inner = %e_outer) -> tensor<?x3xf64> {
      %arg0 = tensor.extract %points[%iv, %jv] : tensor<?x3xf64>
      %i_0 = tensor.extract %correspondences[%iv] : tensor<?xi32>
      %i = arith.index_cast %i_0 : i32 to index
      %vp = tensor.extract %vertex_positions[%i, %jv] : tensor<544x3xf64>
      %0 = arith.subf %arg0, %vp : f64
      %e_next = tensor.insert %0 into %e_inner[%iv, %jv] : tensor<?x3xf64>
      scf.yield %e_next : tensor<?x3xf64>
    }
    scf.yield %err_partial : tensor<?x3xf64>
  }
  return %err : tensor<?x3xf64>
}

func.func @lagrad_hand_objective(
  %theta : tensor<26xf64>,
  %parents: tensor<22xi32>,
  %base_relatives: tensor<22x4x4xf64>,
  %inverse_base_absolutes: tensor<22x4x4xf64>,
  %base_positions: tensor<544x4xf64>,
  %weights: tensor<544x22xf64>,
  %correspondences: tensor<?xi32>,
  %points : tensor<?x3xf64>,
  %g : tensor<?x3xf64>
) -> tensor<26xf64> {
  %dtheta = lagrad.grad @mlir_hand_objective(
    %theta,
    %parents,
    %base_relatives,
    %inverse_base_absolutes,
    %base_positions,
    %weights,
    %correspondences,
    %points,
    %g
  ) {grad_signal} : (
    tensor<26xf64>,
    tensor<22xi32>,
    tensor<22x4x4xf64>,
    tensor<22x4x4xf64>,
    tensor<544x4xf64>,
    tensor<544x22xf64>,
    tensor<?xi32>,
    tensor<?x3xf64>,
    tensor<?x3xf64>
  ) -> tensor<26xf64>
  return %dtheta : tensor<26xf64>
}
