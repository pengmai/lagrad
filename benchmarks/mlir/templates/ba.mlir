func @cross(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
  %cross_space = constant dense<0.0> : tensor<3xf64>
  %res = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0) -> ((d0 + 1) mod 3)>,
        affine_map<(d0) -> ((d0 + 2) mod 3)>,
        affine_map<(d0) -> ((d0 + 2) mod 3)>,
        affine_map<(d0) -> ((d0 + 1) mod 3)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%arg0, %arg1, %arg0, %arg1 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>)
    outs(%cross_space : tensor<3xf64>) {
  ^bb0(%arg2: f64, %arg3: f64, %arg4: f64, %arg5: f64, %arg6: f64):
    %0 = mulf %arg2, %arg3 : f64
    %1 = mulf %arg4, %arg5 : f64
    %2 = subf %0, %1 : f64
    linalg.yield %2 : f64
  } -> tensor<3xf64>
  return %res : tensor<3xf64>
}

func @rodrigues_rotate_point(%rot: tensor<3xf64>, %X: tensor<3xf64>) -> tensor<3xf64> {
  %zero = constant 0.0 : f64
  %one = constant 1.0 : f64
  %w_space = constant dense<0.0> : tensor<3xf64>
  // Square rot, then sum
  %sqtheta_init = constant dense<0.0> : tensor<f64>
  %sqtheta_tensor = linalg.generic
    {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    }
    ins(%rot : tensor<3xf64>)
    outs(%sqtheta_init : tensor<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = mulf %arg0, %arg0 : f64
    %1 = addf %0, %arg1 : f64
    linalg.yield %1 : f64
  } -> tensor<f64>
  %sqtheta = tensor.extract %sqtheta_tensor[] : tensor<f64>

  %cond = cmpf "one", %sqtheta, %zero : f64
  %result = scf.if %cond -> tensor<3xf64> {
    %theta = math.sqrt %sqtheta : f64
    %costheta = math.cos %theta : f64
    %sintheta = math.sin %theta : f64
    %theta_inv = divf %one, %theta : f64

    %w = linalg.generic
      {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      }
      ins(%X : tensor<3xf64>)
      outs(%w_space : tensor<3xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = mulf %theta_inv, %arg0 : f64
      linalg.yield %0 : f64
    } -> tensor<3xf64>

    %w_cross_X = call @cross(%w, %X) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %tmp_0 = constant dense<0.0> : tensor<f64>
    %tmp_1 = linalg.dot ins(%w, %X : tensor<3xf64>, tensor<3xf64>) outs(%tmp_0 : tensor<f64>) -> tensor<f64>
    %tmp_2 = tensor.extract %tmp_1[] : tensor<f64>
    %tmp_3 = subf %one, %costheta : f64
    %tmp = mulf %tmp_2, %tmp_3 : f64

    %w_times_tmp = linalg.generic
      {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      }
      ins(%w : tensor<3xf64>)
      outs(%w_space : tensor<3xf64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = mulf %arg0, %tmp : f64
      linalg.yield %0 : f64
    } -> tensor<3xf64>
    scf.yield %w_times_tmp : tensor<3xf64>
  } else {
    %rot_cross_X = call @cross(%rot, %X) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %X_plus_rot_cross_X = addf %X, %rot_cross_X : tensor<3xf64>
    scf.yield %X_plus_rot_cross_X : tensor<3xf64>
  }
  return %result : tensor<3xf64>
}

func @mlir_compute_reproj_error(
  %cam: tensor<{{n}}x{{nCamParams}}xf64>,
  %X: tensor<{{m}}x3xf64>,
  %w: tensor<{{p}}xf64>,
  %feat: tensor<{{p}}x2xf64>
) -> tensor<2xf64> {
  %cst = constant dense<2.1> : tensor<2xf64>
  return %cst : tensor<2xf64>
}
