module  {
  memref.global "private" constant @__constant_64xf64 : memref<{{n}}xf64> = dense<0.000000e+00>
  func @etrmv_full(%arg0: memref<{{n}}x{{n}}xf64>, %arg1: memref<{{n}}xf64>, %out: memref<{{n}}xf64>) -> f64 {
    linalg.matvec ins(%arg0, %arg1 : memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>) outs(%out : memref<{{n}}xf64>)
    %ret = arith.constant 0.0 : f64
    return %ret : f64
  }

  func @enzyme_trmv_full(%arg0: memref<{{n}}x{{n}}xf64>, %arg1: memref<{{n}}xf64>) -> (memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>) {
    %darg0 = memref.alloc() : memref<{{n}}x{{n}}xf64>
    %darg1 = memref.alloc() : memref<{{n}}xf64>
    %out = memref.alloc() : memref<{{n}}xf64>
    %dout = memref.alloc() : memref<{{n}}xf64>
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    linalg.fill(%zero, %darg0) : f64, memref<{{n}}x{{n}}xf64>
    linalg.fill(%zero, %darg1) : f64, memref<{{n}}xf64>
    linalg.fill(%one, %dout) : f64, memref<{{n}}xf64>
    %f = constant @etrmv_full : (memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
    %df = standalone.diff %f :
      (memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64,
      (memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
    call_indirect %df(%arg0, %darg0, %arg1, %darg1, %out, %dout) : (memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
    return %darg0, %darg1 : memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>
  }

  func @etrmv_tri(%arg0: memref<{{n}}x{{n}}xf64>, %arg1: memref<{{n}}xf64>, %out: memref<{{n}}xf64>) -> f64 {
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant {{n}} : index
    scf.for %arg2 = %c0 to %c64 step %c1 {
      scf.for %arg3 = %c0 to %arg2 step %c1 {
        %1 = memref.load %arg0[%arg2, %arg3] : memref<{{n}}x{{n}}xf64>
        %2 = memref.load %arg1[%arg3] : memref<{{n}}xf64>
        %3 = memref.load %out[%arg2] : memref<{{n}}xf64>
        %4 = arith.mulf %1, %2 : f64
        %5 = arith.addf %3, %4 : f64
        memref.store %5, %out[%arg2] : memref<{{n}}xf64>
      }
    }
    %ret = arith.constant 0.0 : f64
    return %ret : f64
  }

  func @enzyme_trmv_tri(%arg0: memref<{{n}}x{{n}}xf64>, %arg1: memref<{{n}}xf64>) -> (memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>) {
    %darg0 = memref.alloc() : memref<{{n}}x{{n}}xf64>
    %darg1 = memref.alloc() : memref<{{n}}xf64>
    %out = memref.alloc() : memref<{{n}}xf64>
    %dout = memref.alloc() : memref<{{n}}xf64>
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    linalg.fill(%zero, %darg0) : f64, memref<{{n}}x{{n}}xf64>
    linalg.fill(%zero, %darg1) : f64, memref<{{n}}xf64>
    linalg.fill(%one, %dout) : f64, memref<{{n}}xf64>
    %f = constant @etrmv_tri : (memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
    %df = standalone.diff %f :
      (memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64,
      (memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
    call_indirect %df(%arg0, %darg0, %arg1, %darg1, %out, %dout) : (memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
    return %darg0, %darg1 : memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>
  }

  func @etrmv_packed(%arg0: memref<{{tri_size}}xf64>, %arg1: memref<{{n}}xf64>, %out: memref<{{n}}xf64>) -> f64 {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant {{n}} : index
    scf.for %arg2 = %c0 to %c64 step %c1 {
      %2 = arith.addi %arg2, %c1 : index
      scf.for %arg3 = %2 to %c64 step %c1 {
        %idx0 = arith.addi %arg2, %c1 : index
        %idx1 = arith.muli %c2, %c64 : index
        %idx2 = arith.subi %idx1, %idx0 : index
        %idx3 = arith.muli %arg2, %idx2 : index
        %idx4 = arith.divui %idx3, %c2 : index
        %idx5 = arith.subi %arg3, %idx0 : index
        %8 = arith.addi %idx5, %idx4 : index
        %9 = memref.load %arg0[%8] : memref<{{tri_size}}xf64>
        %10 = memref.load %arg1[%arg2] : memref<{{n}}xf64>
        %11 = memref.load %out[%arg3] : memref<{{n}}xf64>
        %12 = arith.mulf %9, %10 : f64
        %13 = arith.addf %12, %11 : f64
        memref.store %13, %out[%arg3] : memref<{{n}}xf64>
      }
    }
    %ret = arith.constant 0.0 : f64
    return %ret : f64
  }

  func @enzyme_trmv_packed(%arg0: memref<{{tri_size}}xf64>, %arg1: memref<{{n}}xf64>) -> (memref<{{tri_size}}xf64>, memref<{{n}}xf64>) {
    %darg0 = memref.alloc() : memref<{{tri_size}}xf64>
    %darg1 = memref.alloc() : memref<{{n}}xf64>
    %out = memref.alloc() : memref<{{n}}xf64>
    %dout = memref.alloc() : memref<{{n}}xf64>
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    linalg.fill(%zero, %darg0) : f64, memref<{{tri_size}}xf64>
    linalg.fill(%zero, %darg1) : f64, memref<{{n}}xf64>
    linalg.fill(%one, %dout) : f64, memref<{{n}}xf64>
    %f = constant @etrmv_packed : (memref<{{tri_size}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
    %df = standalone.diff %f :
      (memref<{{tri_size}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64,
      (memref<{{tri_size}}xf64>, memref<{{tri_size}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
    call_indirect %df(%arg0, %darg0, %arg1, %darg1, %out, %dout) : (memref<{{tri_size}}xf64>, memref<{{tri_size}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
    return %darg0, %darg1 : memref<{{tri_size}}xf64>, memref<{{n}}xf64>
  }
}

