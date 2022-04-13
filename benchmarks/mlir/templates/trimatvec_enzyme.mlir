// {% if method == "full" %}
func @mlir_enzyme_trimatvec_dense_primal(%arg0: memref<{{n}}x{{n}}xf64>, %arg1: memref<{{n}}xf64>, %arg2: memref<{{n}}xf64>) -> f64 {
  %0 = memref.alloc() : memref<{{n}}xf64>
  linalg.copy(%arg2, %0) : memref<{{n}}xf64>, memref<{{n}}xf64>
  linalg.matvec ins(%arg0, %arg1 : memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>) outs(%0 : memref<{{n}}xf64>)

  linalg.copy(%0, %arg2) : memref<{{n}}xf64>, memref<{{n}}xf64>
  memref.dealloc %0 : memref<{{n}}xf64>
  %ret = arith.constant 0.0 : f64
  return %ret : f64
}

func @mlir_enzyme_trimatvec_dense_adjoint(%arg0: memref<{{n}}x{{n}}xf64>, %arg1: memref<{{n}}xf64>, %arg2: memref<{{n}}xf64>) -> (memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>) {
  %tdout = arith.constant dense<1.0> : tensor<{{n}}xf64>
  %zero = arith.constant 0.0 : f64
  %darg0 = memref.alloc() : memref<{{n}}x{{n}}xf64>
  %darg1 = memref.alloc() : memref<{{n}}xf64>
  linalg.fill(%zero, %darg0) : f64, memref<{{n}}x{{n}}xf64>
  linalg.fill(%zero, %darg1) : f64, memref<{{n}}xf64>
  %dout = memref.buffer_cast %tdout : memref<{{n}}xf64>
  %f = constant @mlir_enzyme_trimatvec_dense_primal : (memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  %df = standalone.diff %f : (memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64, (memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  call_indirect %df(%arg0, %darg0, %arg1, %darg1, %arg2, %dout) : (memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  return %darg0, %darg1 : memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>
}

// {% elif method == "tri" %}

func @mlir_enzyme_trimatvec_tri_primal(%arg0: memref<{{n}}x{{n}}xf64>, %arg1: memref<{{n}}xf64>, %arg2: memref<{{n}}xf64>) -> f64 {
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cn = arith.constant {{n}} : index
  %0 = memref.alloc() : memref<{{n}}xf64>
  linalg.fill(%cst, %0) : f64, memref<{{n}}xf64>
  scf.for %arg3 = %c0 to %cn step %c1 {
    scf.for %arg5 = %c0 to %arg3 step %c1 {
      %3 = memref.load %arg0[%arg3, %arg5] : memref<{{n}}x{{n}}xf64>
      %4 = memref.load %arg1[%arg5] : memref<{{n}}xf64>
      %5 = memref.load %0[%arg3] : memref<{{n}}xf64>
      %6 = arith.mulf %3, %4 : f64
      %7 = arith.addf %5, %6 : f64
      memref.store %7, %0[%arg3] : memref<{{n}}xf64>
    }
  }
  linalg.copy(%0, %arg2) : memref<{{n}}xf64>, memref<{{n}}xf64>
  return %cst : f64
}

func @mlir_enzyme_trimatvec_tri_adjoint(%arg0: memref<{{n}}x{{n}}xf64>, %arg1: memref<{{n}}xf64>, %arg2: memref<{{n}}xf64>) -> (memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>) {
  %tdout = arith.constant dense<1.0> : tensor<{{n}}xf64>
  %zero = arith.constant 0.0 : f64
  %darg0 = memref.alloc() : memref<{{n}}x{{n}}xf64>
  %darg1 = memref.alloc() : memref<{{n}}xf64>
  linalg.fill(%zero, %darg0) : f64, memref<{{n}}x{{n}}xf64>
  linalg.fill(%zero, %darg1) : f64, memref<{{n}}xf64>
  %dout = memref.buffer_cast %tdout : memref<{{n}}xf64>
  %f = constant @mlir_enzyme_trimatvec_tri_primal : (memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  %df = standalone.diff %f : (memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64, (memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  call_indirect %df(%arg0, %darg0, %arg1, %darg1, %arg2, %dout) : (memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  return %darg0, %darg1 : memref<{{n}}x{{n}}xf64>, memref<{{n}}xf64>
}
// {% endif %}
