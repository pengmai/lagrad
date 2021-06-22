// Example of a dot product implemented using SCF, not the linalg dialect.

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @print_1d(%arg : memref<?xf32>) {
  %U = memref_cast %arg : memref<?xf32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

func @scfdot(%A : memref<4xf32>, %B : memref<4xf32>) -> f32 {
  %zero = constant 0 : index
  %c1 = constant 1 : index
  %ub = dim %A, %zero : memref<4xf32>
  %sum_0 = constant 0.0 : f32
  %sum = scf.for %iv = %zero to %ub step %c1 iter_args(%sum_iter = %sum_0) -> f32 {
    %t1 = load %A[%iv] : memref<4xf32>
    %t2 = load %B[%iv] : memref<4xf32>
    %both = mulf %t1, %t2 : f32
    %sum_next = addf %sum_iter, %both : f32
    scf.yield %sum_next : f32
  }
  return %sum : f32
}

func @main() {
  %cA = constant dense<[-0.3, 1.4, 2.2, -3.0]> : tensor<4xf32>
  %A = tensor_to_memref %cA : memref<4xf32>
  %dA = alloca() : memref<4xf32>
  %cB = constant dense<[-5.0, 3.4, -10.2, 3.33]> : tensor<4xf32>
  %B = tensor_to_memref %cB : memref<4xf32>
  %dB = alloca() : memref<4xf32>

  // Always remember to zero out the allocated array gradient!
  %zero = constant 0.0 : f32
  linalg.fill(%dB, %zero) : memref<4xf32>, f32

  %f = constant @scfdot : (memref<4xf32>, memref<4xf32>) -> f32
  %df = standalone.diff %f : (memref<4xf32>, memref<4xf32>) -> f32, (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> f32
  call_indirect %df(%A, %dA, %B, %dB) : (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> f32

  %U = memref_cast %dB : memref<4xf32> to memref<?xf32>
  call @print_1d(%U) : (memref<?xf32>) -> ()
  return
}
