func @arrmax(%x: tensor<4xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %max_init = tensor.extract %x[%c0] : tensor<4xf64>
  %max = scf.for %iv = %c1 to %c4 step %c1 iter_args(%max_it = %max_init) -> f64 {
    %ai = tensor.extract %x[%iv] : tensor<4xf64>
    %p = arith.cmpf ogt, %ai, %max_it : f64
    %max_next = select %p, %ai, %max_it : f64
    scf.yield %max_next : f64
  }
  return %max : f64
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %x = arith.constant dense<[-1.2, 5.4, -7.6, -0.8]> : tensor<4xf64>
  // %res = call @arrmax(%x) : (tensor<4xf64>) -> f64
  %f = constant @arrmax : (tensor<4xf64>) -> f64
  %df = standalone.grad %f : (tensor<4xf64>) -> f64, (tensor<4xf64>) -> tensor<4xf64>
  %res = call_indirect %df(%x) : (tensor<4xf64>) -> tensor<4xf64>
  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
