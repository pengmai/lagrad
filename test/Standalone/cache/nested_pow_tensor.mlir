func.func @tensor_pow(%x: tensor<4xf64>, %n: index) -> tensor<4xf64> {
  %r_init = arith.constant dense<1.0> : tensor<4xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %res = scf.for %iv = %c0 to %n step %c1 iter_args(%r_outer = %r_init) -> tensor<4xf64> {
    %r_outer_next = scf.for %jv = %c0 to %c3 step %c1 iter_args(%r = %r_outer) -> tensor<4xf64> {
      %r_next = arith.mulf %r, %x : tensor<4xf64>
      scf.yield %r_next : tensor<4xf64>
    }
    scf.yield %r_outer_next : tensor<4xf64>
  }
  return %res : tensor<4xf64>
}

func.func private @printMemrefF64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func.func @main() {
  %x = arith.constant dense<1.3> : tensor<4xf64>
  %n = arith.constant 4 : index
  %f = constant @tensor_pow : (tensor<4xf64>, index) -> tensor<4xf64>
  %df = standalone.grad %f : (tensor<4xf64>, index) -> tensor<4xf64>, (tensor<4xf64>, index) -> tensor<4xf64>
  %res = call_indirect %df(%x, %n) : (tensor<4xf64>, index) -> tensor<4xf64>
  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  return
}
