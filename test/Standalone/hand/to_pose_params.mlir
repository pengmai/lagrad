func @mto_pose_params(%theta: tensor<26xf64>) -> tensor<25x3xf64> {
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %ones = arith.constant dense<1.0> : tensor<3xf64>
  %pose_params_space = linalg.init_tensor [25, 3] : tensor<25x3xf64>
  %pose_params_init = linalg.fill(%zero, %pose_params_space) : f64, tensor<25x3xf64> -> tensor<25x3xf64>
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

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }
func private @print_memref_i64(memref<*xi64>) attributes { llvm.emit_c_interface }

func @main() {
  %theta = arith.constant dense<[
    1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
    14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.
  ]> : tensor<26xf64>
  %pose_params = call @mto_pose_params(%theta) : (tensor<26xf64>) -> tensor<25x3xf64>
  %U = tensor.cast %pose_params : tensor<25x3xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()

  %f = constant @mto_pose_params : (tensor<26xf64>) -> tensor<25x3xf64>
  %df = standalone.grad %f : (tensor<26xf64>) -> tensor<25x3xf64>, (tensor<26xf64>) -> tensor<26xf64>
  %res = call_indirect %df(%theta) : (tensor<26xf64>) -> tensor<26xf64>
  %U0 = tensor.cast %res : tensor<26xf64> to tensor<*xf64>
  call @print_memref_f64(%U0) : (tensor<*xf64>) -> ()
  return
}
