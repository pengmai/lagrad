func @prod(%t: tensor<4xf64>) -> f64 {
  %init = arith.constant 1.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%it = %init) -> f64 {
    %el = tensor.extract %t[%iv] : tensor<4xf64>
    %it_next = arith.mulf %it, %el : f64
    scf.yield %it_next : f64
  }
  return %res : f64
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %f = constant @prod : (tensor<4xf64>) -> f64
  %df = standalone.grad %f : (tensor<4xf64>) -> f64, (tensor<4xf64>) -> tensor<4xf64>
  %A = arith.constant dense<[1., 2., 3., 4.]> : tensor<4xf64>
  %res = call_indirect %df(%A) : (tensor<4xf64>) -> tensor<4xf64>
  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
