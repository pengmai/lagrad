// A very simple example of MLIR that operates on an array.
func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @print_0d(%arg : memref<f32>) {
  %U = memref_cast %arg :  memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

func @print_1d(%arg : memref<4xf32>) {
  %U = memref_cast %arg : memref<4xf32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

func @sum(%arr : memref<4xf32>) -> f32 {
  %sum_0 = constant 0.0 : f32
  // This is a special constant that should always be zero.
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  // This corresponds to the first element of the array.
  %lb = constant 0 : index
  %ub = dim %arr, %c0 : memref<4xf32>
  %sum = scf.for %iv = %lb to %ub step %c1 iter_args(%sum_iter = %sum_0) -> f32 {
    %t = load %arr[%iv] : memref<4xf32>
    %sum_next = addf %sum_iter, %t : f32
    scf.yield %sum_next : f32
  }
  return %sum : f32
}

func @main() {
  %cst = constant 2.0 : f32
  %zero = constant 0.0 : f32
  %arr = alloca() : memref<4xf32>
  %darr = alloca() : memref<4xf32>
  linalg.fill(%arr, %cst) : memref<4xf32>, f32
  linalg.fill(%darr, %zero) : memref<4xf32>, f32

  %fp = constant @sum : (memref<4xf32>) -> f32
  %df = standalone.diff %fp : (memref<4xf32>) -> f32, (memref<4xf32>, memref<4xf32>) -> f32
  %dres = call_indirect %df(%arr, %darr) : (memref<4xf32>, memref<4xf32>) -> f32
  call @print_1d(%darr) : (memref<4xf32>) -> ()
  return
}