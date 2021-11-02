#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]

func @generic_matmul(%A : memref<4x4xf32>, %B : memref<4x4xf32>, %C : memref<4x4xf32>) -> f32 {
  // linalg.generic
  //   { indexing_maps = #matmul_accesses, iterator_types = ["parallel", "parallel", "reduction"] }
  //   ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
  //   outs(%C : memref<4x4xf32>) {
  // ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
  //   %0 = arith.mulf %arg0, %arg1 : f32
  //   %1 = arith.addf %0, %arg2 : f32
  //   linalg.yield %1 : f32
  // }
  linalg.matmul ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>) outs(%C : memref<4x4xf32>)

  %cst = arith.constant 0.0 : f32
  return %cst : f32
}

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @main() -> i64 {
  %source_A = arith.constant dense<[
    [ 1.,  2.,  3.,  4.],
    [ 5.,  6.,  7.,  8.],
    [ 9., 10., 11., 12.],
    [13., 14., 15., 16.]
  ]> : tensor<4x4xf32>
  %source_B = arith.constant dense<[
    [ 1.,  2.,  3.,  4.],
    [-5., -6.,  7.,  8.],
    [ 9., 10.,-11., 12.],
    [13., 14., 15., 16.]
  ]> : tensor<4x4xf32>
  %source_C = arith.constant dense<1.0> : tensor<4x4xf32>
  %zero = arith.constant 0.0 : f32

  %A = memref.buffer_cast %source_A : memref<4x4xf32>
  %dA = memref.alloca() : memref<4x4xf32>
  linalg.fill(%zero, %dA) : f32, memref<4x4xf32>
  %B = memref.buffer_cast %source_B : memref<4x4xf32>
  %dB = memref.alloca() : memref<4x4xf32>
  linalg.fill(%zero, %dB) : f32, memref<4x4xf32>
  %C = memref.buffer_cast %source_C : memref<4x4xf32>

  %f = constant @generic_matmul : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> f32
  %df = standalone.diff %f : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> f32, (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> f32
  call_indirect %df(%A, %dA, %B, %dB, %C, %C) : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> f32

  %U = memref.cast %dA : memref<4x4xf32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()

  %exit = arith.constant 0 : i64
  return %exit : i64
}
