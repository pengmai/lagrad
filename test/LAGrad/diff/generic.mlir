#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]

func @generic_matmul(%A : memref<4x4xf64>, %B : memref<4x4xf64>, %C : memref<4x4xf64>) -> f64 {
  linalg.generic
    { indexing_maps = #matmul_accesses, iterator_types = ["parallel", "parallel", "reduction"] }
    ins(%A, %B : memref<4x4xf64>, memref<4x4xf64>)
    outs(%C : memref<4x4xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    linalg.yield %1 : f64
  }

  %cst = arith.constant 0.0 : f64
  return %cst : f64
}

func private @print_memref_f64(memref<*xf64>) attributes { llvm.emit_c_interface }

func @main() -> i64 {
  %source_A = arith.constant dense<[
    [ 1.,  2.,  3.,  4.],
    [ 5.,  6.,  7.,  8.],
    [ 9., 10., 11., 12.],
    [13., 14., 15., 16.]
  ]> : tensor<4x4xf64>
  %source_B = arith.constant dense<[
    [ 1.,  2.,  3.,  4.],
    [-5., -6.,  7.,  8.],
    [ 9., 10.,-11., 12.],
    [13., 14., 15., 16.]
  ]> : tensor<4x4xf64>
  %source_C = arith.constant dense<1.0> : tensor<4x4xf64>
  %zero = arith.constant 0.0 : f64

  %A = memref.buffer_cast %source_A : memref<4x4xf64>
  %dA = memref.alloca() : memref<4x4xf64>
  linalg.fill(%zero, %dA) : f64, memref<4x4xf64>
  %B = memref.buffer_cast %source_B : memref<4x4xf64>
  %dB = memref.alloca() : memref<4x4xf64>
  linalg.fill(%zero, %dB) : f64, memref<4x4xf64>
  %out = memref.alloca() : memref<4x4xf64>
  linalg.fill(%zero, %out) : f64, memref<4x4xf64>
  %C = memref.buffer_cast %source_C : memref<4x4xf64>

  %f = constant @generic_matmul : (memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>) -> f64
  %df = lagrad.diff %f : (memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>) -> f64, (memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>) -> f64
  call_indirect %df(%A, %dA, %B, %dB, %out, %C) : (memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>) -> f64

  %U = memref.cast %dA : memref<4x4xf64> to memref<*xf64>
  call @print_memref_f64(%U) : (memref<*xf64>) -> ()

  %exit = arith.constant 0 : i64
  return %exit : i64
}
