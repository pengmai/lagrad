func @emdot(%x: memref<{{n}}xf64>, %y: memref<{{n}}xf64>) -> f64 {
  %out = memref.alloca() : memref<f64>
  linalg.dot ins(%x, %y : memref<{{n}}xf64>, memref<{{n}}xf64>) outs(%out: memref<f64>)
  %outval = memref.load %out[] : memref<f64>
  return %outval : f64
}

func @enzyme_mlir_dot(%x: memref<{{n}}xf64>, %y: memref<{{n}}xf64>) -> memref<{{n}}xf64> {
  %zero = arith.constant 0.0 : f64
  %dx = memref.alloc() : memref<{{n}}xf64>
  linalg.fill(%zero, %dx) : f64, memref<{{n}}xf64>

  %f = constant @emdot : (memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  %df = standalone.diff %f {const = [1]} : (memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64, (memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  call_indirect %df(%x, %dx, %y) : (memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  return %dx : memref<{{n}}xf64>
}
