#map = affine_map<(d0) -> (d0)>
func @emvecadd(%x: memref<{{n}}xf64>, %y: memref<{{n}}xf64>, %out: memref<{{n}}xf64>) -> f64 {
  linalg.generic
    {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel"]
    }
    ins(%x, %y : memref<{{n}}xf64>, memref<{{n}}xf64>)
    outs(%out : memref<{{n}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.addf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  }
  %ret = arith.constant 0.0 : f64
  return %ret : f64
}

func @enzyme_mlir_vecadd(%x: memref<{{n}}xf64>, %y: memref<{{n}}xf64>, %g: memref<{{n}}xf64>) -> memref<{{n}}xf64> {
  %zero = arith.constant 0.0 : f64
  %dx = memref.alloc() : memref<{{n}}xf64>
  linalg.fill(%zero, %dx) : f64, memref<{{n}}xf64>
  %out = memref.alloc() : memref<{{n}}xf64>
  linalg.fill(%zero, %out) : f64, memref<{{n}}xf64>

  %f = constant @emvecadd : (memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  %df = standalone.diff %f {const = [1]}:
    (memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64,
    (memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  call_indirect %df(%x, %dx, %y, %out, %g) : (memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>, memref<{{n}}xf64>) -> f64
  // memref.dealloc %out : memref<{{n}}xf64>
  return %dx : memref<{{n}}xf64>
}
