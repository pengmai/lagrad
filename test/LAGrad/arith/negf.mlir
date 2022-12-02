func.func @negf(%arg0: f64) -> f64 {
  %1 = arith.negf %arg0 : f64
  return %1 : f64
}

func.func private @printMemrefF64(tensor<*xf64>) attributes {llvm.emit_c_interface}

func.func @main() {
  %arg = arith.constant -3.2 : f64
  %res = lagrad.grad @negf(%arg) : (f64) -> f64
  %t = tensor.empty() : tensor<f64>
  %t0 = tensor.insert %res into %t[] : tensor<f64>
  %t1 = tensor.cast %t0 : tensor<f64> to tensor<*xf64>
  call @printMemrefF64(%t1) : (tensor<*xf64>) -> ()
  return
}
