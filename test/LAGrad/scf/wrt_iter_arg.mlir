// Inspired by LSTMs, this function computes an adjoint w.r.t. a value that is carried through
// the loop iter args.

func @wrt_iter_arg(%state_outer: tensor<2x3xf64>) -> tensor<3xf64> {
  %cst = arith.constant dense<[1., 2., 3.]> : tensor<3xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %res:2 = scf.for %iv = %c0 to %c2 step %c1 iter_args(%x = %cst, %state = %state_outer) -> (tensor<3xf64>, tensor<2x3xf64>) {
    %slice = tensor.extract_slice %state[%iv, 0] [1, 3] [1, 1] : tensor<2x3xf64> to tensor<3xf64>
    %mul = arith.mulf %slice, %x : tensor<3xf64>
    %new_state = tensor.insert_slice %mul into %state[%iv, 0] [1, 3] [1, 1] : tensor<3xf64> into tensor<2x3xf64>
    %add = arith.addf %mul, %x : tensor<3xf64>
    scf.yield %add, %new_state : tensor<3xf64>, tensor<2x3xf64>
  }
  return %res#0 : tensor<3xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %state_outer = arith.constant dense<[[2., 2., 2.], [3., 2., 2.]]> : tensor<2x3xf64>
  %f = constant @wrt_iter_arg : (tensor<2x3xf64>) -> tensor<3xf64>
  %df = standalone.grad %f {of = [0]} : (tensor<2x3xf64>) -> tensor<3xf64>, (tensor<2x3xf64>) -> tensor<2x3xf64>
  %res = call_indirect %df(%state_outer) : (tensor<2x3xf64>) -> tensor<2x3xf64>
  %U = tensor.cast %res : tensor<2x3xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
