// Example of a dot product implemented using SCF, not the linalg dialect.

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @print_1d(%arg : memref<?xf32>) {
  %U = memref.cast %arg : memref<?xf32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

func @scfdot(%A : memref<4xf32>, %B : memref<4xf32>) -> f32 {
  %zero = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ub = memref.dim %A, %zero : memref<4xf32>
  %sum_0 = arith.constant 0.0 : f32
  %sum = scf.for %iv = %zero to %ub step %c1 iter_args(%sum_iter = %sum_0) -> f32 {
    %t1 = memref.load %A[%iv] : memref<4xf32>
    %t2 = memref.load %B[%iv] : memref<4xf32>
    %both = arith.mulf %t1, %t2 : f32
    %sum_next = arith.addf %sum_iter, %both : f32
    scf.yield %sum_next : f32
  }
  return %sum : f32
}

func @main() -> i64 {
  %cA = arith.constant dense<[-0.3, 1.4, 2.2, -3.0]> : tensor<4xf32>
  %A = bufferization.to_memref %cA : memref<4xf32>
  %dA = memref.alloca() : memref<4xf32>
  %cB = arith.constant dense<[-5.0, 3.4, -10.2, 3.33]> : tensor<4xf32>
  %B = bufferization.to_memref %cB : memref<4xf32>
  %dB = memref.alloca() : memref<4xf32>

  // Always remember to zero out the allocated array gradient!
  %zero = arith.constant 0.0 : f32
  linalg.fill(%zero, %dB) : f32, memref<4xf32>

  %f = constant @scfdot : (memref<4xf32>, memref<4xf32>) -> f32
  %df = standalone.diff %f : (memref<4xf32>, memref<4xf32>) -> f32, (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> f32
  call_indirect %df(%A, %dA, %B, %dB) : (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> f32

  %U = memref.cast %dB : memref<4xf32> to memref<?xf32>
  call @print_1d(%U) : (memref<?xf32>) -> ()
  %exit = arith.constant 0 : i64
  return %exit : i64
}
