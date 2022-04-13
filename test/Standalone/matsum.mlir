// A simple program that operates on a 2d array:
// Compute the sum of its elements.
func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @matsum(%arr : memref<4x4xf32>) -> f32 {
  %res_0 = arith.constant 0.0 : f32
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %4 = arith.constant 4 : index
  %res = scf.for %iv = %0 to %4 step %1 iter_args(%sum_iter = %res_0) -> f32 {
    %partial_res = scf.for %jv = %0 to %4 step %1 iter_args(%sum_inner = %sum_iter) -> f32 {
      %t = memref.load %arr[%iv, %jv] : memref<4x4xf32>
      %sum_next = arith.addf %sum_inner, %t : f32
      scf.yield %sum_next : f32
    }
    scf.yield %partial_res : f32
  }

  return %res : f32
}

func @main() -> i64 {
  %cA = arith.constant dense<[
    [0.3, 2.3, -1.2, 3.0],
    [5.6, -8.7, -30.2, 1.1],
    [3.2, 9.0, 9.0, -9.5],
    [3.4, 7.8, 10.2, -11.2]
  ]> : tensor<4x4xf32>
  %A = bufferization.to_memref %cA : memref<4x4xf32>
  %dA = memref.alloca() : memref<4x4xf32>
  %0 = arith.constant 0.0 : f32
  linalg.fill(%0, %dA) : f32, memref<4x4xf32>

  %f = constant @matsum : (memref<4x4xf32>) -> f32
  %df = standalone.diff %f : (memref<4x4xf32>) -> f32, (memref<4x4xf32>, memref<4x4xf32>) -> f32
  call_indirect %df(%A, %dA) : (memref<4x4xf32>, memref<4x4xf32>) -> f32

  %U = memref.cast %dA : memref<4x4xf32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  %exit = arith.constant 0 : i64
  return %exit : i64
}
