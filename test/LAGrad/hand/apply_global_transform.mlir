func @mangle_axis_to_rotation_matrix(%angle_axis: tensor<3xf64>) -> tensor<3x3xf64> {
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
  %res_space = linalg.init_tensor [3, 3] : tensor<3x3xf64>
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

func @mapply_global_transform(%pose_params: tensor<25x3xf64>, %positions: tensor<544x3xf64>) -> tensor<544x3xf64> {
  %angle_axis = tensor.extract_slice %pose_params[0, 0] [1, 3] [1, 1] : tensor<25x3xf64> to tensor<3xf64>
  %pose_slice = tensor.extract_slice %pose_params[1, 0] [1, 3] [1, 1] : tensor<25x3xf64> to tensor<3xf64>
  %pose_slice_2 = tensor.extract_slice %pose_params[2, 0] [1, 3] [1, 1] : tensor<25x3xf64> to tensor<3xf64>
  %R_0 = call @mangle_axis_to_rotation_matrix(%angle_axis) : (tensor<3xf64>) -> tensor<3x3xf64>
  %zero = arith.constant dense<0.0> : tensor<3x3xf64>

  %R = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%R_0, %pose_slice : tensor<3x3xf64>, tensor<3xf64>)
    outs(%zero : tensor<3x3xf64>) {
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

  %positions_new = linalg.generic
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
  return %positions_new : tensor<544x3xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %A = tensor.generate {
  ^bb0(%i: index, %j: index):
    %0 = arith.muli %i, %c3 : index
    %1 = arith.addi %0, %j : index
    %2 = arith.addi %1, %c1 : index
    %3 = arith.index_cast %2 : index to i64
    %4 = arith.sitofp %3 : i64 to f64
    tensor.yield %4 : f64
  } : tensor<25x3xf64>
  %B = tensor.generate {
  ^bb0(%i: index, %j: index):
    %0 = arith.muli %i, %c3 : index
    %1 = arith.addi %0, %j : index
    %2 = arith.addi %1, %c1 : index
    %3 = arith.index_cast %2 : index to i64
    %4 = arith.sitofp %3 : i64 to f64
    tensor.yield %4 : f64
  } : tensor<544x3xf64>
  %res = lagrad.grad @mapply_global_transform(%A, %B) : (tensor<25x3xf64>, tensor<544x3xf64>) -> tensor<25x3xf64>
  %U = tensor.cast %res : tensor<25x3xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}