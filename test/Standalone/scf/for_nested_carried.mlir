func @f(%state: tensor<4xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f64
  %res:2 = scf.for %iv = %c0 to %c3 step %c1 iter_args(%fl = %zero, %state_it = %state) -> (f64, tensor<4xf64>) {
    %el = tensor.extract %state_it[%iv] : tensor<4xf64>
    %fl2 = arith.addf %fl, %el : f64
    scf.yield %fl2, %state_it : f64, tensor<4xf64>
  }
  return %res#0 : f64
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %A = arith.constant dense<[0.1, 0.2, 0.3, 0.4]> : tensor<4xf64>
  %f = constant @f : (tensor<4xf64>) -> f64
  %df = standalone.grad %f {of = [0]} : (tensor<4xf64>) -> f64, (tensor<4xf64>) -> tensor<4xf64>
  %res = call_indirect %df(%A) : (tensor<4xf64>) -> tensor<4xf64>
  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
