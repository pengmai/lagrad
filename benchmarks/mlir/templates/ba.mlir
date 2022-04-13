#map_1d_id = affine_map<(d0) -> (d0)>

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @mlir_compute_reproj_error(
  %cam: tensor<{{nCamParams}}xf64>,
  %X: tensor<3xf64>,
  %w: f64,
  %feat: tensor<2xf64>,
  %out: tensor<2xf64>
) -> tensor<2xf64> {
  %cst = arith.constant dense<0.0> : tensor<2xf64>

  %cam_rot = tensor.extract_slice %cam[{{rot_idx}}] [3] [1] : tensor<{{nCamParams}}xf64> to tensor<3xf64>
  %cam_c = tensor.extract_slice %cam[{{c_idx}}] [3] [1] : tensor<{{nCamParams}}xf64> to tensor<3xf64>
  %cam_rad = tensor.extract_slice %cam[{{rad_idx}}] [2] [1] : tensor<{{nCamParams}}xf64> to tensor<2xf64>

  %f_idx = arith.constant {{f_idx}} : index
  %cam_f = tensor.extract %cam[%f_idx] : tensor<{{nCamParams}}xf64>
  %cam_x0 = tensor.extract_slice %cam[{{x0_idx}}] [2] [1] : tensor<{{nCamParams}}xf64> to tensor<2xf64>

  %Xsubcam = arith.subf %X, %cam_c : tensor<3xf64>
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %w_space = arith.constant dense<0.0> : tensor<3xf64>
  %tmp_0 = arith.constant dense<0.0> : tensor<f64>
  // Square rot, then sum
  %sqtheta_init = arith.constant dense<0.0> : tensor<f64>
  %X_costheta_space = arith.constant dense<0.0> : tensor<3xf64>
  %w_cross_X_space = arith.constant dense<0.0> : tensor<3xf64>
  %sqtheta_tensor = linalg.generic
    {
      indexing_maps = [#map_1d_id, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    }
    ins(%cam_rot : tensor<3xf64>)
    outs(%sqtheta_init : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = arith.mulf %arg0, %arg0 : f64
    %1 = arith.addf %0, %arg1 : f64
    linalg.yield %1 : f64
  } -> tensor<f64>

  %sqtheta = tensor.extract %sqtheta_tensor[] : tensor<f64>

  %final = linalg.generic
    {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    outs(%out : tensor<2xf64>) {
  ^bb0(%arg0: f64):
    linalg.yield %sqtheta : f64
  } -> tensor<2xf64>
  return %final : tensor<2xf64>

  // %Xcam_space = linalg.init_tensor [3] : tensor<3xf64>
  // %cond = arith.cmpf "one", %sqtheta, %zero : f64
  // %Xcam = scf.if %cond -> tensor<3xf64> {
  //   %theta = math.sqrt %sqtheta : f64
  //   %costheta = math.cos %theta : f64
  //   %sintheta = math.sin %theta : f64
  //   %theta_inv = arith.divf %one, %theta : f64

  //   %w0 = linalg.generic
  //     {
  //       indexing_maps = [#map_1d_id, #map_1d_id],
  //       iterator_types = ["parallel"]
  //     }
  //     ins(%cam_rot : tensor<3xf64>)
  //     outs(%Xcam_space : tensor<3xf64>) {
  //   ^bb0(%arg2: f64, %arg3: f64):
  //     %0 = arith.mulf %arg2, %theta_inv : f64
  //     linalg.yield %0 : f64
  //   } -> tensor<3xf64>
  //   %U = tensor.cast %w0 : tensor<3xf64> to tensor<*xf64>
  //   call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  //   scf.yield %w0 : tensor<3xf64>
  //   // %w_cross_X = linalg.generic
  //   //   {
  //   //     indexing_maps = [
  //   //       affine_map<(d0) -> ((d0 + 1) mod 3)>,
  //   //       affine_map<(d0) -> ((d0 + 2) mod 3)>,
  //   //       affine_map<(d0) -> ((d0 + 2) mod 3)>,
  //   //       affine_map<(d0) -> ((d0 + 1) mod 3)>,
  //   //       #map_1d_id
  //   //     ],
  //   //     iterator_types = ["parallel"]
  //   //   }
  //   //   ins(%w0, %Xsubcam, %w0, %Xsubcam : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>)
  //   //   outs(%w_space : tensor<3xf64>) {
  //   // ^bb0(%arg2: f64, %arg3: f64, %arg4: f64, %arg5: f64, %arg6: f64):
  //   //   %0 = arith.mulf %arg2, %arg3 : f64
  //   //   %1 = arith.mulf %arg4, %arg5 : f64
  //   //   %2 = arith.subf %0, %1 : f64
  //   //   linalg.yield %2 : f64
  //   // } -> tensor<3xf64>
  //   // scf.yield %w_cross_X : tensor<3xf64>

  //   // %tmp_1 = linalg.dot ins(%w0, %Xsubcam : tensor<3xf64>, tensor<3xf64>) outs(%tmp_0 : tensor<f64>) -> tensor<f64>
  //   // %tmp_2 = tensor.extract %tmp_1[] : tensor<f64>
  //   // %tmp_3 = arith.subf %one, %costheta : f64
  //   // %tmp = arith.mulf %tmp_2, %tmp_3 : f64

  //   // %X_costheta = linalg.generic
  //   //   {
  //   //     indexing_maps = [#map_1d_id, #map_1d_id],
  //   //     iterator_types = ["parallel"]
  //   //   }
  //   //   ins(%Xsubcam : tensor<3xf64>)
  //   //   outs(%X_costheta_space : tensor<3xf64>) {
  //   // ^bb0(%arg2: f64, %arg3: f64):
  //   //   %0 = arith.mulf %arg2, %costheta : f64
  //   //   linalg.yield %0 : f64
  //   // } -> tensor<3xf64>
  //   // // scf.yield %X_costheta : tensor<3xf64>

  //   // %w_cross_X_sintheta = linalg.generic
  //   //   {
  //   //     indexing_maps = [#map_1d_id, #map_1d_id],
  //   //     iterator_types = ["parallel"]
  //   //   }
  //   //   ins(%w_cross_X : tensor<3xf64>)
  //   //   outs(%X_costheta_space : tensor<3xf64>) {
  //   // ^bb0(%arg2: f64, %arg3: f64):
  //   //   %0 = arith.mulf %arg2, %sintheta : f64
  //   //   linalg.yield %0 : f64
  //   // } -> tensor<3xf64>
  //   // // scf.yield %w_cross_X_sintheta : tensor<3xf64>

  //   // %w_times_tmp = linalg.generic
  //   //   {
  //   //     indexing_maps = [#map_1d_id, #map_1d_id],
  //   //     iterator_types = ["parallel"]
  //   //   }
  //   //   ins(%w0 : tensor<3xf64>)
  //   //   outs(%X_costheta_space : tensor<3xf64>) {
  //   // ^bb0(%arg2: f64, %arg3: f64):
  //   //   %0 = arith.mulf %arg2, %tmp : f64
  //   //   linalg.yield %0 : f64
  //   // } -> tensor<3xf64>

  //   // %out_0 = arith.addf %X_costheta, %w_cross_X_sintheta : tensor<3xf64>
  //   // // %out_1 = arith.addf %out_0, %w_times_tmp : tensor<3xf64>
  //   // %out_1 = linalg.generic
  //   //   {
  //   //     indexing_maps = [#map_1d_id, #map_1d_id, #map_1d_id],
  //   //     iterator_types = ["parallel"]
  //   //   }
  //   //   ins(%out_0, %w_times_tmp : tensor<3xf64>, tensor<3xf64>)
  //   //   outs(%Xcam_space : tensor<3xf64>) {
  //   // ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
  //   //   %0 = arith.addf %arg2, %arg3 : f64
  //   //   linalg.yield %0 : f64
  //   // } -> tensor<3xf64>
  //   // scf.yield %out_1 : tensor<3xf64>
  // } else {
  //   // %rot_cross_X = linalg.generic
  //   //   {
  //   //     indexing_maps = [
  //   //       affine_map<(d0) -> ((d0 + 1) mod 3)>,
  //   //       affine_map<(d0) -> ((d0 + 2) mod 3)>,
  //   //       affine_map<(d0) -> ((d0 + 2) mod 3)>,
  //   //       affine_map<(d0) -> ((d0 + 1) mod 3)>,
  //   //       #map_1d_id
  //   //     ],
  //   //     iterator_types = ["parallel"]
  //   //   }
  //   //   ins(%cam_rot, %Xsubcam, %cam_rot, %Xsubcam : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>)
  //   //   outs(%cross_space : tensor<3xf64>) {
  //   // ^bb0(%arg2: f64, %arg3: f64, %arg4: f64, %arg5: f64, %arg6: f64):
  //   //   %0 = arith.mulf %arg2, %arg3 : f64
  //   //   %1 = arith.mulf %arg4, %arg5 : f64
  //   //   %2 = arith.subf %0, %1 : f64
  //   //   linalg.yield %2 : f64
  //   // } -> tensor<3xf64>
  //   %yielded = linalg.fill(%zero, %Xcam_space) : f64, tensor<3xf64> -> tensor<3xf64>
  //   %U = tensor.cast %yielded : tensor<3xf64> to tensor<*xf64>
  //   call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  //   scf.yield %yielded : tensor<3xf64>
  //   // %X_plus_rot_cross_X = arith.addf %Xsubcam, %rot_cross_X : tensor<3xf64>
  //   // %X_plus_rot_cross_X = linalg.generic
  //   //   {
  //   //     indexing_maps = [#map_1d_id, #map_1d_id],
  //   //     iterator_types = ["parallel"]
  //   //   }
  //   //   ins(%Xsubcam : tensor<3xf64>)
  //   //   outs(%rot_cross_X : tensor<3xf64>) {
  //   // ^bb0(%arg2: f64, %arg3: f64):
  //   //   %0 = arith.addf %arg2, %arg3 : f64
  //   //   linalg.yield %0 : f64
  //   // } -> tensor<3xf64>
  //   // scf.yield %X_plus_rot_cross_X : tensor<3xf64>
  // }

  // %Xcam_start = tensor.extract_slice %Xcam[0] [2] [1] : tensor<3xf64> to tensor<2xf64>
  // %final = linalg.generic
  //   {
  //     indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]
  //   }
  //   ins(%Xcam_start : tensor<2xf64>)
  //   outs(%out : tensor<2xf64>) {
  // ^bb0(%arg2: f64, %arg3: f64):
  //   linalg.yield %arg2 : f64
  // } -> tensor<2xf64>
  // return %final : tensor<2xf64>
  // %twoi = arith.constant 2 : index
  // %Xcam_end = tensor.extract %Xcam[%twoi] : tensor<3xf64>
  // %Xcam_div_space = arith.constant dense<0.0> : tensor<2xf64>
  // %Xcam_div = linalg.generic
  //   {indexing_maps = [#map_1d_id, #map_1d_id], iterator_types = ["parallel"]}
  //   ins(%Xcam_start : tensor<2xf64>)
  //   outs(%Xcam_div_space : tensor<2xf64>) {
  // ^bb0(%arg0: f64, %arg1: f64):
  //   %0 = arith.divf %arg0, %Xcam_end : f64
  //   linalg.yield %0 : f64
  // } -> tensor<2xf64>
  // %rsq_init = arith.constant dense<0.0> : tensor<f64>
  // %rsq_tensor = linalg.generic
  //   {
  //     indexing_maps = [#map_1d_id, affine_map<(d0) -> ()>],
  //     iterator_types = ["reduction"]
  //   }
  //   ins(%Xcam_div : tensor<2xf64>)
  //   outs(%rsq_init : tensor<f64>) {
  // ^bb0(%arg0: f64, %arg1: f64):
  //   %0 = arith.mulf %arg0, %arg0 : f64
  //   %1 = arith.addf %0, %arg1 : f64
  //   linalg.yield %1 : f64
  // } -> tensor<f64>
  // %rsq = tensor.extract %rsq_tensor[] : tensor<f64>

  // %zeroi = arith.constant 0 : index
  // %onei = arith.constant 1 : index
  // %onef = arith.constant 1.0 : f64

  // %rp0 = tensor.extract %cam_rad[%zeroi] : tensor<2xf64>
  // %rp1 = tensor.extract %cam_rad[%onei] : tensor<2xf64>

  // %L0 = arith.mulf %rp0, %rsq : f64
  // %L1 = arith.mulf %rp1, %rsq : f64
  // %L2 = arith.mulf %L1, %rsq : f64
  // %L3 = arith.addf %L0, %L2 : f64
  // %L4 = arith.addf %onef, %L3 : f64

  // %distorted = linalg.generic
  //   {
  //     indexing_maps = [#map_1d_id, #map_1d_id],
  //     iterator_types = ["parallel"]
  //   }
  //   ins(%Xcam_div : tensor<2xf64>)
  //   outs(%out : tensor<2xf64>) {
  // ^bb0(%arg0: f64, %arg1: f64):
  //   %0 = arith.mulf %arg0, %L4 : f64
  //   linalg.yield %0 : f64
  // } -> tensor<2xf64>

  // %proj = linalg.generic
  //   {indexing_maps = [#map_1d_id, #map_1d_id], iterator_types = ["parallel"]}
  //   ins(%cam_x0 : tensor<2xf64>)
  //   outs(%distorted : tensor<2xf64>) {
  // ^bb0(%arg0: f64, %arg1: f64):
  //   %0 = arith.mulf %arg1, %cam_f : f64
  //   %1 = arith.addf %0, %arg0 : f64
  //   linalg.yield %1 : f64
  // } -> tensor<2xf64>
  // // %proj = arith.addf %proj0, %cam_x0 : tensor<2xf64>

  // %err = linalg.generic
  //   {indexing_maps = [#map_1d_id, #map_1d_id], iterator_types = ["parallel"]}
  //   ins(%feat : tensor<2xf64>)
  //   outs(%proj : tensor<2xf64>) {
  // ^bb0(%arg0: f64, %arg1: f64):
  //   %0 = arith.subf %arg1, %arg0 : f64
  //   %1 = arith.mulf %w, %0 : f64
  //   linalg.yield %1 : f64
  // } -> tensor<2xf64>
  // return %err : tensor<2xf64>
}

func @mlir_compute_zach_weight_error(%w: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %0 = arith.mulf %w, %w : f64
  %1 = arith.subf %one, %0 : f64
  return %1 : f64
}

func @lagrad_compute_reproj_error(
  %cam: tensor<{{nCamParams}}xf64>,
  %dcam: tensor<{{nCamParams}}xf64>,
  %X: tensor<3xf64>,
  %dX: tensor<3xf64>,
  %w: f64,
  %dw: f64,
  %feat: tensor<2xf64>,
  %g: tensor<2xf64>
) -> (tensor<{{nCamParams}}xf64>, tensor<3xf64>, f64) {
  %out = linalg.init_tensor [2] : tensor<2xf64>
  %f = constant @mlir_compute_reproj_error : (tensor<{{nCamParams}}xf64>, tensor<3xf64>, f64, tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  // %df = standalone.grad %f {of = [0, 1, 2], grad_signal = true, dps = true} : (
  //   tensor<{{nCamParams}}xf64>,
  //   tensor<3xf64>,
  //   f64,
  //   tensor<2xf64>,
  //   tensor<2xf64> // out
  // ) -> tensor<2xf64>, (
  //   tensor<{{nCamParams}}xf64>, // cam
  //   tensor<3xf64>,              // X
  //   f64,                        // w
  //   tensor<2xf64>,              // feat
  //   tensor<2xf64>,              // out
  //   tensor<{{nCamParams}}xf64>, // dcam
  //   tensor<3xf64>,              // dX
  //   f64,                        // dw
  //   tensor<2xf64>               // g
  // ) -> (tensor<{{nCamParams}}xf64>, tensor<3xf64>, f64)
  // %res:3 = call_indirect %df(%cam, %X, %w, %feat, %out, %dcam, %dX, %dw, %g) : (
  //   tensor<{{nCamParams}}xf64>,
  //   tensor<3xf64>,
  //   f64,
  //   tensor<2xf64>,
  //   tensor<2xf64>,
  //   tensor<{{nCamParams}}xf64>,
  //   tensor<3xf64>,
  //   f64,
  //   tensor<2xf64>
  // ) -> (tensor<{{nCamParams}}xf64>, tensor<3xf64>, f64)
  // return %res#0, %res#1, %res#2 : tensor<{{nCamParams}}xf64>, tensor<3xf64>, f64
  return %dcam, %dX, %dw : tensor<{{nCamParams}}xf64>, tensor<3xf64>, f64
}

func @lagrad_compute_w_error(%w: f64) -> f64 {
  %f = constant @mlir_compute_zach_weight_error : (f64) -> f64
  %df = standalone.grad %f {of = [0]} : (f64) -> f64, (f64) -> f64
  %res = call_indirect %df(%w) : (f64) -> f64
  return %res : f64
}
