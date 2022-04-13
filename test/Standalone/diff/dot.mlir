func @dot(%A : memref<4xf32>, %B : memref<4xf32>, %C : memref<f32> {linalg.inplaceable = true}) -> f32 {
  linalg.dot ins(%A, %B : memref<4xf32>, memref<4xf32>) outs(%C : memref<f32>)
  %cst = arith.constant 0.0 : f32
  return %cst : f32
}

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @main() -> i64 {
  %source_A = arith.constant dense<[1., 2., 3., 4.]> : tensor<4xf32>
  %source_B = arith.constant dense<[9., 10., -11., 12.]> : tensor<4xf32>
  %source_C = arith.constant dense<1.0> : tensor<f32>
  %zero = arith.constant 0.0 : f32

  %A = bufferization.to_memref %source_A : memref<4xf32>
  %dA = memref.alloca() : memref<4xf32>
  linalg.fill(%zero, %dA) : f32, memref<4xf32>
  %B = bufferization.to_memref %source_B : memref<4xf32>
  %dB = memref.alloca() : memref<4xf32>
  linalg.fill(%zero, %dB) : f32, memref<4xf32>
  %C = bufferization.to_memref %source_C : memref<f32>

  %f = constant @dot : (memref<4xf32>, memref<4xf32>, memref<f32>) -> f32
  %df = standalone.diff %f : (memref<4xf32>, memref<4xf32>, memref<f32>) -> f32, (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<f32>, memref<f32>) -> f32
  call_indirect %df(%A, %dA, %B, %dB, %C, %C) : (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<f32>, memref<f32>) -> f32

  %U = memref.cast %dA : memref<4xf32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  %exit = arith.constant 0 : i64
  return %exit : i64
}
