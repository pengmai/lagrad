#map_1d_id = affine_map<(d0) -> (d0)>

func @cross(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
  %cross_space = arith.constant dense<0.0> : tensor<3xf64>
  %res = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0) -> ((d0 + 1) mod 3)>,
        affine_map<(d0) -> ((d0 + 2) mod 3)>,
        affine_map<(d0) -> ((d0 + 2) mod 3)>,
        affine_map<(d0) -> ((d0 + 1) mod 3)>,
        #map_1d_id
      ],
      iterator_types = ["parallel"]
    }
    ins(%arg0, %arg1, %arg0, %arg1 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>)
    outs(%cross_space : tensor<3xf64>) {
  ^bb0(%arg2: f64, %arg3: f64, %arg4: f64, %arg5: f64, %arg6: f64):
    %0 = arith.mulf %arg2, %arg3 : f64
    %1 = arith.mulf %arg4, %arg5 : f64
    %2 = arith.subf %0, %1 : f64
    linalg.yield %2 : f64
  } -> tensor<3xf64>
  return %res : tensor<3xf64>
}

func @scalvec(%arg0: f64, %arg1: tensor<3xf64>) -> tensor<3xf64> {
  %out_init = arith.constant dense<0.0> : tensor<3xf64>
  %out = linalg.generic
    {
      indexing_maps = [#map_1d_id, #map_1d_id],
      iterator_types = ["parallel"]
    }
    ins(%arg1 : tensor<3xf64>)
    outs(%out_init : tensor<3xf64>) {
  ^bb0(%arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg2, %arg0 : f64
    linalg.yield %0 : f64
  } -> tensor<3xf64>
  return %out : tensor<3xf64>
}

// There should be a way to make this dynamically shaped, but I'd need to know
// how to safely generate the initial value.
func @scalvec2(%arg0: f64, %arg1: tensor<2xf64>) -> tensor<2xf64> {
  %out_init = arith.constant dense<0.0> : tensor<2xf64>
  %out = linalg.generic
    {
      indexing_maps = [#map_1d_id, #map_1d_id],
      iterator_types = ["parallel"]
    }
    ins(%arg1 : tensor<2xf64>)
    outs(%out_init : tensor<2xf64>) {
  ^bb0(%arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg2, %arg0 : f64
    linalg.yield %0 : f64
  } -> tensor<2xf64>
  return %out : tensor<2xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @rodrigues_rotate_point(%rot: tensor<3xf64>, %X: tensor<3xf64>) -> tensor<3xf64> {
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %w_space = arith.constant dense<0.0> : tensor<3xf64>
  %tmp_0 = arith.constant dense<0.0> : tensor<f64>
  // Square rot, then sum
  %sqtheta_init = arith.constant dense<0.0> : tensor<f64>
  %X_costheta_space = linalg.init_tensor [3] : tensor<3xf64>

  %sqtheta_tensor = linalg.generic
    {
      indexing_maps = [#map_1d_id, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    }
    ins(%rot : tensor<3xf64>)
    outs(%sqtheta_init : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = arith.mulf %arg0, %arg0 : f64
    %1 = arith.addf %0, %arg1 : f64
    linalg.yield %1 : f64
  } -> tensor<f64>
  %sqtheta = tensor.extract %sqtheta_tensor[] : tensor<f64>

  %cond = arith.cmpf "one", %sqtheta, %zero : f64
  %result = scf.if %cond -> tensor<3xf64> {
    %theta = math.sqrt %sqtheta : f64
    %costheta = math.cos %theta : f64
    // %sintheta = math.sin %theta : f64
    // %theta_inv = arith.divf %one, %theta : f64

    // %w = linalg.generic
    //   {
    //     indexing_maps = [#map_1d_id, #map_1d_id],
    //     iterator_types = ["parallel"]
    //   }
    //   ins(%rot : tensor<3xf64>)
    //   outs(%w_space : tensor<3xf64>) {
    // ^bb0(%arg2: f64, %arg3: f64):
    //   %0 = arith.mulf %arg2, %theta_inv : f64
    //   linalg.yield %0 : f64
    // } -> tensor<3xf64>
    // %w_cross_X = call @cross(%w, %X) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>

    // %tmp_1 = linalg.dot ins(%w, %X : tensor<3xf64>, tensor<3xf64>) outs(%tmp_0 : tensor<f64>) -> tensor<f64>
    // %tmp_2 = tensor.extract %tmp_1[] : tensor<f64>
    // %tmp_3 = arith.subf %one, %costheta : f64
    // %tmp = arith.mulf %tmp_2, %tmp_3 : f64

    // %X_costheta = call @scalvec(%costheta, %X) : (f64, tensor<3xf64>) -> tensor<3xf64>
    %X_costheta = linalg.generic
      {
        indexing_maps = [#map_1d_id, #map_1d_id],
        iterator_types = ["parallel"]
      }
      ins(%X : tensor<3xf64>)
      outs(%X_costheta_space : tensor<3xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):
      %0 = arith.mulf %arg2, %costheta : f64
      linalg.yield %0 : f64
    } -> tensor<3xf64>
    scf.yield %X_costheta : tensor<3xf64>
    // %w_cross_X_sintheta = call @scalvec(%sintheta, %w_cross_X) : (f64, tensor<3xf64>) -> tensor<3xf64>
    // %w_times_tmp = call @scalvec(%tmp, %w) : (f64, tensor<3xf64>) -> tensor<3xf64>

    // %out_0 = arith.addf %X_costheta, %w_cross_X_sintheta : tensor<3xf64>
    // %out_1 = arith.addf %out_0, %w_times_tmp : tensor<3xf64>

    // scf.yield %out_1 : tensor<3xf64>
  } else {
    // %rot_cross_X = call @cross(%rot, %X) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    // %X_plus_rot_cross_X = arith.addf %X, %rot_cross_X : tensor<3xf64>
    // scf.yield %X_plus_rot_cross_X : tensor<3xf64>
    scf.yield %rot : tensor<3xf64>
  }
  return %result : tensor<3xf64>
}

// func @radial_distort(%rad_params: tensor<2xf64>, %proj: tensor<2xf64>) -> tensor<2xf64> {
//   %rsq_init = arith.constant dense<0.0> : tensor<f64>
//   %rsq_tensor = linalg.generic
//     {
//       indexing_maps = [#map_1d_id, affine_map<(d0) -> ()>],
//       iterator_types = ["reduction"]
//     }
//     ins(%proj : tensor<2xf64>)
//     outs(%rsq_init : tensor<f64>) {
//   ^bb0(%arg0: f64, %arg1: f64):
//     %0 = arith.mulf %arg0, %arg0 : f64
//     %1 = arith.addf %0, %arg1 : f64
//     linalg.yield %1 : f64
//   } -> tensor<f64>
//   %rsq = tensor.extract %rsq_tensor[] : tensor<f64>

//   %zeroi = arith.constant 0 : index
//   %onei = arith.constant 1 : index
//   %onef = arith.constant 1.0 : f64

//   %rp0 = tensor.extract %rad_params[%zeroi] : tensor<2xf64>
//   %rp1 = tensor.extract %rad_params[%onei] : tensor<2xf64>

//   %L0 = arith.mulf %rp0, %rsq : f64
//   %L1 = arith.mulf %rp1, %rsq : f64
//   %L2 = arith.mulf %L1, %rsq : f64
//   %L3 = arith.addf %L0, %L2 : f64
//   %L4 = arith.addf %onef, %L3 : f64

//   %out_init = arith.constant dense<0.0> : tensor<2xf64>
//   %out = linalg.generic
//     {
//       indexing_maps = [#map_1d_id, #map_1d_id],
//       iterator_types = ["parallel"]
//     }
//     ins(%proj : tensor<2xf64>)
//     outs(%out_init : tensor<2xf64>) {
//   ^bb0(%arg0: f64, %arg1: f64):
//     %0 = arith.mulf %arg0, %L4 : f64
//     linalg.yield %0 : f64
//   } -> tensor<2xf64>
//   return %out : tensor<2xf64>
// }

// func @project(%cam: tensor<{{nCamParams}}xf64>, %X: tensor<3xf64>) -> tensor<2xf64> {
//   %cam_rot = tensor.extract_slice %cam[{{rot_idx}}] [3] [1] : tensor<{{nCamParams}}xf64> to tensor<3xf64>
//   %cam_c = tensor.extract_slice %cam[{{c_idx}}] [3] [1] : tensor<{{nCamParams}}xf64> to tensor<3xf64>
//   %cam_rad = tensor.extract_slice %cam[{{rad_idx}}] [2] [1] : tensor<{{nCamParams}}xf64> to tensor<2xf64>

//   %f_idx = arith.constant {{f_idx}} : index
//   %cam_f = tensor.extract %cam[%f_idx] : tensor<{{nCamParams}}xf64>
//   %cam_x0 = tensor.extract_slice %cam[{{x0_idx}}] [2] [1] : tensor<{{nCamParams}}xf64> to tensor<2xf64>

//   %Xsubcam = arith.subf %X, %cam_c : tensor<3xf64>
//   // %Xcam = call @rodrigues_rotate_point(%cam_rot, %Xsubcam) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>

//   %cst = arith.constant dense<0.0> : tensor<2xf64>
//   return %cst : tensor<2xf64>
//   // %Xcam_start = tensor.extract_slice %Xcam[0] [2] [1] : tensor<3xf64> to tensor<2xf64>
//   // %twoi = arith.constant 2 : index
//   // %Xcam_end = tensor.extract %Xcam[%twoi] : tensor<3xf64>
//   // %Xcam_div_space = arith.constant dense<0.0> : tensor<2xf64>
//   // %Xcam_div = linalg.generic
//   //   {indexing_maps = [#map_1d_id, #map_1d_id], iterator_types = ["parallel"]}
//   //   ins(%Xcam_start : tensor<2xf64>)
//   //   outs(%Xcam_div_space : tensor<2xf64>) {
//   // ^bb0(%arg0: f64, %arg1: f64):
//   //   %0 = arith.divf %arg0, %Xcam_end : f64
//   //   linalg.yield %0 : f64
//   // } -> tensor<2xf64>

//   // %distorted = call @radial_distort(%cam_rad, %Xcam_div) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
//   // %res0 = call @scalvec2(%cam_f, %distorted) : (f64, tensor<2xf64>) -> tensor<2xf64>
//   // %res = arith.addf %res0, %cam_x0 : tensor<2xf64>
//   // return %res : tensor<2xf64>
// }

func @grad_rrp(%rot: tensor<3xf64>, %X: tensor<3xf64>) -> tensor<3xf64> {
  %f = constant @rodrigues_rotate_point : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  %df = standalone.grad %f : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>, (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  %res = call_indirect %df(%rot, %X) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  return %res : tensor<3xf64>
}

// func @grad_project(%cam: tensor<{{nCamParams}}xf64>, %X: tensor<3xf64>) -> tensor<{{nCamParams}}xf64> {
//   %f = constant @project : (tensor<{{nCamParams}}xf64>, tensor<3xf64>) -> tensor<2xf64>
//   // %df = standalone.grad %f : (tensor<{{nCamParams}}xf64>, tensor<3xf64>) -> tensor<2xf64>, (tensor<{{nCamParams}}xf64>, tensor<3xf64>) -> tensor<{{nCamParams}}xf64>
//   // %res = call_indirect %df(%cam, %X) : (tensor<{{nCamParams}}xf64>, tensor<3xf64>) -> tensor<{{nCamParams}}xf64>
//   %res = arith.constant dense<0.0> : tensor<{{nCamParams}}xf64>

//   %f1 = constant @rodrigues_rotate_point : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
//   %df = standalone.grad %f1 : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>, (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
//   return %res : tensor<{{nCamParams}}xf64>
// }

// func @mlir_compute_reproj_error(
//   %cam: tensor<{{n}}x{{nCamParams}}xf64>,
//   %X: tensor<{{m}}x3xf64>,
//   %w: tensor<{{p}}xf64>,
//   %feat: tensor<{{p}}x2xf64>
// ) -> tensor<2xf64> {
//   %cst = arith.constant dense<2.1> : tensor<2xf64>
//   return %cst : tensor<2xf64>
// }
