func.func @bmm(%A: tensor<4x3x5xf64>, %B: tensor<4x5x3xf64>) -> tensor<4x3x3xf64> {
  %init = arith.constant dense<0.0> : tensor<4x3x3xf64>
  %out = linalg.batch_matmul ins(%A, %B : tensor<4x3x5xf64>, tensor<4x5x3xf64>) outs(%init : tensor<4x3x3xf64>) -> tensor<4x3x3xf64>
  return %out : tensor<4x3x3xf64>
}

func.func private @printMemrefF64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func.func @main() {
  %A = arith.constant dense<1.2> : tensor<4x3x5xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c7 = arith.constant 7 : index
  %c15 = arith.constant 15 : index
  %B = tensor.generate {
  ^bb0(%i: index, %j: index, %k: index):
    %0 = arith.muli %i, %c15 : index
    %1 = arith.muli %j, %c3 : index
    %2 = arith.addi %0, %1 : index
    %3 = arith.addi %2, %k : index
    %4 = arith.addi %3, %c1 : index
    %p_0 = arith.remsi %4, %c7 : index
    %p = arith.cmpi "eq", %p_0, %c0 : index
    %5 = arith.index_cast %4 : index to i64
    %6 = arith.sitofp %5 : i64 to f64
    %7 = arith.negf %6 : f64
    %8 = arith.select %p, %7, %6 : f64
    tensor.yield %8 : f64
  } : tensor<4x5x3xf64>

  %f = constant @bmm : (tensor<4x3x5xf64>, tensor<4x5x3xf64>) -> tensor<4x3x3xf64>
  %df = standalone.grad %f {of = [0]} : (tensor<4x3x5xf64>, tensor<4x5x3xf64>) -> tensor<4x3x3xf64>, (tensor<4x3x5xf64>, tensor<4x5x3xf64>) -> tensor<4x3x5xf64>
  %res = call_indirect %df(%A, %B) : (tensor<4x3x5xf64>, tensor<4x5x3xf64>) -> tensor<4x3x5xf64>
  %U = tensor.cast %res : tensor<4x3x5xf64> to tensor<*xf64>
  call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  return
}
