#map = affine_map<(d0) -> (d0)>
func @emvecscal(%x: memref<{{n}}xf64>, %scal: memref<f64>, %out: memref<{{n}}xf64>) -> f64 {
  linalg.generic
    {
      indexing_maps = [#map, affine_map<(d0) -> ()>, #map],
      iterator_types = ["parallel"]
    }
    ins(%x, %scal : memref<{{n}}xf64>, memref<f64>)
    outs(%out : memref<{{n}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  }
  %ret = arith.constant 0.0 : f64
  return %ret : f64
}

func @enzyme_mlir_vecscal(%x: memref<{{n}}xf64>, %scal: memref<f64>, %g: memref<{{n}}xf64>) -> f64 {
  %zero = arith.constant 0.0 : f64
  %dscal = memref.alloca() : memref<f64>
  %out = memref.alloc() : memref<{{n}}xf64>
  memref.store %zero, %dscal[] : memref<f64>
  linalg.fill(%zero, %out) : f64, memref<{{n}}xf64>

  %f = constant @emvecscal : (memref<{{n}}xf64>, memref<f64>, memref<{{n}}xf64>) -> f64
  %df = standalone.diff %f {const = [0]} :
    (memref<{{n}}xf64>, memref<f64>, memref<{{n}}xf64>) -> f64,
    (memref<{{n}}xf64>, memref<f64>, memref<f64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  call_indirect %df(%x, %scal, %dscal, %out, %g) : (memref<{{n}}xf64>, memref<f64>, memref<f64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64

  %res = memref.load %dscal[] : memref<f64>
  return %res : f64
}
