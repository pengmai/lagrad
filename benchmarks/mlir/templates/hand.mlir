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
  %pp_1 = scf.for %iv = %c0 to %c3 step %c1 iter_args(%ppiv = %pose_params_init) -> (tensor<{{nbones + 3}}x3xf64>) {
    %th0 = tensor.extract %theta[%iv] : tensor<{{nbones}}xf64>
    %th3_idx = arith.addi %iv, %c3 : index
    %th1 = tensor.extract %theta[%th3_idx] : tensor<{{nbones}}xf64>
    %ppiv_0 = tensor.insert %th0 into %ppiv[%c0, %iv] : tensor<{{nbones + 3}}x3xf64>
    %ppiv_1 = tensor.insert %one into %ppiv_0[%c1, %iv] : tensor<{{nbones + 3}}x3xf64>
    %ppiv_2 = tensor.insert %th1 into %ppiv_1[%c2, %iv] : tensor<{{nbones + 3}}x3xf64>
    scf.yield %ppiv_2 : tensor<{{nbones + 3}}x3xf64>
  }

  return %pp_1 : tensor<{{nbones + 3}}x3xf64>
}

func @dtopose_params(%theta: tensor<{{nbones}}xf64>) -> tensor<{{nbones}}xf64> {
  %f = constant @mto_pose_params : (tensor<{{nbones}}xf64>) -> tensor<{{nbones + 3}}x3xf64>
  %df = standalone.grad %f : (tensor<{{nbones}}xf64>) -> tensor<{{nbones + 3}}x3xf64>, (tensor<{{nbones}}xf64>) -> tensor<{{nbones}}xf64>
  %res = call_indirect %df(%theta) : (tensor<{{nbones}}xf64>) -> tensor<{{nbones}}xf64>
  return %res : tensor<{{nbones}}xf64>
}
