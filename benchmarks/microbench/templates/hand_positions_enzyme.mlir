#map0 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s0 + d1 * s2 + s1)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map4 = affine_map<(d0, d1) -> (d0 * 4 + d1)>
#map5 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d0)>
module  {
  memref.global "private" constant @__constant_544x4xf64 : memref<544x4xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_544x3xf64 : memref<544x3xf64> = dense<0.000000e+00>
  func @en_calc_positions(%arg0: memref<22x4x4xf64>, %arg1: memref<544x4xf64>, %arg2: memref<544x22xf64>, %out : memref<544x3xf64>) -> f64 {
    %0 = memref.alloc() : memref<544x3xf64>
    // %0 = memref.get_global @__constant_544x3xf64 : memref<544x3xf64>
    %cst = arith.constant 0.000000e+00 : f64
    linalg.fill(%cst, %0) : f64, memref<544x3xf64>
    %1 = memref.get_global @__constant_544x4xf64 : memref<544x4xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c22 = arith.constant 22 : index
    %2 = scf.for %arg3 = %c0 to %c22 step %c1 iter_args(%arg4 = %0) -> (memref<544x3xf64>) {
      %3 = memref.subview %arg0[%arg3, 0, 0] [1, 4, 4] [1, 1, 1] : memref<22x4x4xf64> to memref<4x4xf64, #map0>
      %4 = memref.cast %3 : memref<4x4xf64, #map0> to memref<4x4xf64>
      %5 = memref.alloc() : memref<544x4xf64>
      linalg.copy(%1, %5) : memref<544x4xf64>, memref<544x4xf64> 
      linalg.generic {doc = "Column-major matrix multiplication", indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%4, %arg1 : memref<4x4xf64>, memref<544x4xf64>) outs(%5 : memref<544x4xf64>) {
      ^bb0(%arg5: f64, %arg6: f64, %arg7: f64):  // no predecessors
        %11 = arith.mulf %arg5, %arg6 : f64
        %12 = arith.addf %11, %arg7 : f64
        linalg.yield %12 : f64
      }
      %6 = memref.alloc() : memref<544x3xf64>
      %7 = memref.subview %5[0, 0] [544, 3] [1, 1] : memref<544x4xf64> to memref<544x3xf64, #map4>
      linalg.copy(%7, %6) : memref<544x3xf64, #map4>, memref<544x3xf64>
      %8 = memref.subview %arg2[0, %arg3] [544, 1] [1, 1] : memref<544x22xf64> to memref<544xf64, #map5>
      %9 = memref.cast %8 : memref<544xf64, #map5> to memref<544xf64>
      %10 = memref.alloc() : memref<544x3xf64>
      linalg.copy(%arg4, %10) : memref<544x3xf64>, memref<544x3xf64> 
      linalg.generic {indexing_maps = [#map6, #map7, #map6], iterator_types = ["parallel", "parallel"]} ins(%6, %9 : memref<544x3xf64>, memref<544xf64>) outs(%10 : memref<544x3xf64>) {
      ^bb0(%arg5: f64, %arg6: f64, %arg7: f64):  // no predecessors
        %11 = arith.mulf %arg5, %arg6 : f64
        %12 = arith.addf %11, %arg7 : f64
        linalg.yield %12 : f64
      }
      scf.yield %10 : memref<544x3xf64>
      // scf.yield %arg4 : memref<544x3xf64>
    }
    linalg.copy(%2, %out) : memref<544x3xf64>, memref<544x3xf64>
    %ret = arith.constant 0.0 : f64
    return %ret : f64
  }

  func @den_positions(%arg0: memref<22x4x4xf64>, %arg1: memref<544x4xf64>, %arg2: memref<544x22xf64>) -> memref<22x4x4xf64> {
    %darg = memref.alloc() : memref<22x4x4xf64>
    %out = memref.alloc() : memref<544x3xf64>
    %dout = memref.alloc() : memref<544x3xf64>
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    linalg.fill(%zero, %darg) : f64, memref<22x4x4xf64>
    linalg.fill(%zero, %out) : f64, memref<544x3xf64>
    linalg.fill(%one, %dout) : f64, memref<544x3xf64>
    %f = constant @en_calc_positions : (memref<22x4x4xf64>, memref<544x4xf64>, memref<544x22xf64>, memref<544x3xf64>) -> f64
    %df = standalone.diff %f {const = [1, 2]} : (memref<22x4x4xf64>, memref<544x4xf64>, memref<544x22xf64>, memref<544x3xf64>) -> f64, (memref<22x4x4xf64>, memref<22x4x4xf64>, memref<544x4xf64>, memref<544x22xf64>, memref<544x3xf64>, memref<544x3xf64>) -> f64
    call_indirect %df(%arg0, %darg, %arg1, %arg2, %out, %dout) : (memref<22x4x4xf64>, memref<22x4x4xf64>, memref<544x4xf64>, memref<544x22xf64>, memref<544x3xf64>, memref<544x3xf64>) -> f64
    memref.dealloc %out : memref<544x3xf64>
    memref.dealloc %dout : memref<544x3xf64>
    return %darg : memref<22x4x4xf64>
  }
}
