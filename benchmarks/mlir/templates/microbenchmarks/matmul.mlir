func @emmatmul(%A: memref<{{n}}x{{n}}xf64>, %B: memref<{{n}}x{{n}}xf64>, %out: memref<{{n}}x{{n}}xf64>) -> f64 {
  linalg.matmul ins(%A, %B : memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>) outs(%out : memref<{{n}}x{{n}}xf64>)
  %zero = arith.constant 0.0 : f64
  return %zero : f64
}

func @enzyme_mlir_matmul(%A: memref<{{n}}x{{n}}xf64>, %B: memref<{{n}}x{{n}}xf64>) -> memref<{{n}}x{{n}}xf64> {
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %dA = memref.alloc() : memref<{{n}}x{{n}}xf64>
  %out = memref.alloc() : memref<{{n}}x{{n}}xf64>
  %dout = memref.alloc() : memref<{{n}}x{{n}}xf64>
  linalg.fill(%zero, %dA) : f64, memref<{{n}}x{{n}}xf64>
  linalg.fill(%zero, %out) : f64, memref<{{n}}x{{n}}xf64>
  linalg.fill(%one, %dout) : f64, memref<{{n}}x{{n}}xf64>

  %f = constant @emmatmul : (memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>) -> f64
  %df = standalone.diff %f {const = [1]} :
    (memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>) -> f64,
    (memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>) -> f64
  call_indirect %df(%A, %dA, %B, %out, %dout) : (memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>, memref<{{n}}x{{n}}xf64>) -> f64
  memref.dealloc %out : memref<{{n}}x{{n}}xf64>
  memref.dealloc %dout : memref<{{n}}x{{n}}xf64>
  return %dA : memref<{{n}}x{{n}}xf64>
}
