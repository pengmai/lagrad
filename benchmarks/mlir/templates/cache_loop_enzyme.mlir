// The primal was taken from cache_loop.mlir, bufferized, then modified for compatibility
// with Enzyme.
#map0 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map1 = affine_map<(d0) -> (d0)>
func @ecache_loop(%arg0: memref<{{n}}x{{d}}xf64>, %parents: memref<{{n}}xi32>, %out:  memref<{{n}}x{{d}}xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cneg1 = arith.constant -1 : index
  %cn = arith.constant {{n}} : index
  %1 = memref.alloc() : memref<{{d}}xf64>
  scf.for %arg1 = %c0 to %cn step %c1 {
    %parent_i = memref.load %parents[%arg1] : memref<{{n}}xi32>
    %parent_idx = arith.index_cast %parent_i : i32 to index
    %3 = arith.cmpi eq, %parent_idx, %cneg1 : index
    %4 = memref.subview %arg0[%arg1, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
    %5 = memref.cast %4 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
    %6 = scf.if %3 -> (memref<{{d}}xf64>) {
      scf.yield %5 : memref<{{d}}xf64>
    } else {
      %9 = memref.subview %out[%parent_idx, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
      %10 = memref.cast %9 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%10, %5 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%1 : memref<{{d}}xf64>) {
      ^bb0(%arg3: f64, %arg4: f64, %arg5: f64):  // no predecessors
        %11 = arith.mulf %arg3, %arg4 : f64
        linalg.yield %11 : f64
      }
      scf.yield %1 : memref<{{d}}xf64>
    }
    %7 = memref.subview %out[%arg1, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
    linalg.copy(%6, %7) : memref<{{d}}xf64>, memref<{{d}}xf64, #map0>
  }
  memref.dealloc %1 : memref<{{d}}xf64>

  %ret = arith.constant 0.0 : f64
  return %ret : f64
}

func @enzyme_cache_loop(%A: memref<{{n}}x{{d}}xf64>, %dA: memref<{{n}}x{{d}}xf64>, %parents: memref<{{n}}xi32>, %out: memref<{{n}}x{{d}}xf64>, %dout: memref<{{n}}x{{d}}xf64>) {
  %f = constant @ecache_loop :  (memref<{{n}}x{{d}}xf64>, memref<{{n}}xi32>, memref<{{n}}x{{d}}xf64>) -> f64
  %df = standalone.diff %f {const = [1]}: (memref<{{n}}x{{d}}xf64>, memref<{{n}}xi32>, memref<{{n}}x{{d}}xf64>) -> f64, (memref<{{n}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>, memref<{{n}}xi32>, memref<{{n}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>) -> f64
  call_indirect %df(%A, %dA, %parents, %out, %dout) : (memref<{{n}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>, memref<{{n}}xi32>, memref<{{n}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>) -> f64
  return
}
