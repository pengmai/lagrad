// A very simple example of MLIR that operates on an array.
func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @print_0d(%arg : memref<f32>) {
  %U = memref.cast %arg :  memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

func @print_1d(%arg : memref<4xf32>) {
  %U = memref.cast %arg : memref<4xf32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

func @sum(%arr : memref<4xf32>) -> f32 {
  %sum_0 = arith.constant 0.0 : f32
  // This is a special constant that should always be zero.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // This corresponds to the first element of the array.
  %lb = arith.constant 0 : index
  %ub = memref.dim %arr, %c0 : memref<4xf32>
  %sum = scf.for %iv = %lb to %ub step %c1 iter_args(%sum_iter = %sum_0) -> f32 {
    %t = memref.load %arr[%iv] : memref<4xf32>
    %sum_next = arith.addf %sum_iter, %t : f32
    scf.yield %sum_next : f32
  }
  return %sum : f32
}

func @main() -> i64 {
  %cst = arith.constant 2.0 : f32
  %zero = arith.constant 0.0 : f32
  %arr = memref.alloca() : memref<4xf32>
  %darr = memref.alloca() : memref<4xf32>
  linalg.fill(%cst, %arr) : f32, memref<4xf32>
  linalg.fill(%zero, %darr) : f32, memref<4xf32>

  %fp = constant @sum : (memref<4xf32>) -> f32
  %df = lagrad.diff %fp : (memref<4xf32>) -> f32, (memref<4xf32>, memref<4xf32>) -> f32
  %dres = call_indirect %df(%arr, %darr) : (memref<4xf32>, memref<4xf32>) -> f32
  call @print_1d(%darr) : (memref<4xf32>) -> ()
  %exit = arith.constant 0 : i64
  return %exit : i64
}
