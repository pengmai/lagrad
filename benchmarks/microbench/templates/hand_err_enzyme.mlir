func @en_hand_err(%arg0: memref<544x3xf64>, %arg1: memref<100x3xf64>, %arg2: memref<100xi32>, %out: memref<100x3xf64>) -> f64 {
  %0 = memref.alloc() : memref<100x3xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c100 = arith.constant 100 : index
  %1 = scf.for %arg3 = %c0 to %c100 step %c1 iter_args(%arg4 = %0) -> (memref<100x3xf64>) {
    %2 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (memref<100x3xf64>) {
      %3 = memref.load %arg1[%arg3, %arg5] : memref<100x3xf64>
      %4 = memref.load %arg2[%arg3] : memref<100xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = memref.load %arg0[%5, %arg5] : memref<544x3xf64>
      %7 = arith.subf %3, %6 : f64
      memref.store %7, %arg6[%arg3, %arg5] : memref<100x3xf64>
      scf.yield %arg6 : memref<100x3xf64>
    }
    scf.yield %2 : memref<100x3xf64>
  }
  linalg.copy(%1, %out) : memref<100x3xf64>, memref<100x3xf64>
  %ret = arith.constant 0.0 : f64
  return %ret : f64
}

func @den_hand_err(%arg0: memref<544x3xf64>, %arg1: memref<100x3xf64>, %arg2: memref<100xi32>) -> memref<544x3xf64> {
  %err = memref.alloc() : memref<100x3xf64>
  %derr = memref.alloc() : memref<100x3xf64>
  %darg0 = memref.alloc() : memref<544x3xf64>
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  linalg.fill(%zero, %err) : f64, memref<100x3xf64>
  linalg.fill(%zero, %darg0) : f64, memref<544x3xf64>
  linalg.fill(%one, %derr) : f64, memref<100x3xf64>

  %f = constant @en_hand_err : (memref<544x3xf64>, memref<100x3xf64>, memref<100xi32>, memref<100x3xf64>) -> f64
  %df = standalone.diff %f {const = [1, 2]} : (memref<544x3xf64>, memref<100x3xf64>, memref<100xi32>, memref<100x3xf64>) -> f64, (memref<544x3xf64>, memref<544x3xf64>, memref<100x3xf64>, memref<100xi32>, memref<100x3xf64>, memref<100x3xf64>) -> f64
  call_indirect %df(%arg0, %darg0, %arg1, %arg2, %err, %derr) : (memref<544x3xf64>, memref<544x3xf64>, memref<100x3xf64>, memref<100xi32>, memref<100x3xf64>, memref<100x3xf64>) -> f64
  memref.dealloc %err : memref<100x3xf64>
  memref.dealloc %derr : memref<100x3xf64>
  return %darg0 : memref<544x3xf64>
}
