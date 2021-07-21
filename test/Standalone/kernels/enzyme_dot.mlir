func @scfdot(%A : memref<131072xf32>, %B : memref<131072xf32>) -> f32 {
  %zero = constant 0 : index
  %c1 = constant 1 : index
  %ub = dim %A, %zero : memref<131072xf32>
  %sum_0 = constant 0.0 : f32
  %sum = scf.for %iv = %zero to %ub step %c1 iter_args(%sum_iter = %sum_0) -> f32 {
    %t1 = load %A[%iv] : memref<131072xf32>
    %t2 = load %B[%iv] : memref<131072xf32>
    %both = mulf %t1, %t2 : f32
    %sum_next = addf %sum_iter, %both : f32
    scf.yield %sum_next : f32
  }
  return %sum : f32
}

func @enzyme_dot(%a : memref<131072xf32>, %b : memref<131072xf32>) -> memref<*xf32> {
  %da = alloc() : memref<131072xf32>
  %db = alloca() : memref<131072xf32>
  %cst = constant 0.0 : f32
  linalg.fill(%da, %cst) : memref<131072xf32>, f32
  linalg.fill(%db, %cst) : memref<131072xf32>, f32
  %f = constant @scfdot : (memref<131072xf32>, memref<131072xf32>) -> f32
  %df = standalone.diff %f : (memref<131072xf32>, memref<131072xf32>) -> f32, (memref<131072xf32>, memref<131072xf32>, memref<131072xf32>, memref<131072xf32>) -> f32
  call_indirect %df(%a, %da, %b, %db) : (memref<131072xf32>, memref<131072xf32>, memref<131072xf32>, memref<131072xf32>) -> f32
  %casted = memref_cast %da : memref<131072xf32> to memref<*xf32>
  return %casted : memref<*xf32>
}
