#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map9 = affine_map<(d0) -> (d0)>
#map10 = affine_map<(d0, d1) -> (d1)>
#map11 = affine_map<(d0) -> ()>
#map12 = affine_map<(d0, d1, d2) -> (d0)>
#map13 = affine_map<() -> ()>
module  {
  memref.global "private" constant @__constant_xf64_0 : memref<f64> = dense<1.000000e+03>
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_{{n}}xf64_0 : memref<{{n}}xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_{{n}}xf64 : memref<{{n}}xf64> = dense<-1.000000e+09>
  memref.global "private" constant @__constant_{{n}}x{{k}}xf64 : memref<{{n}}x{{k}}xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_{{n}}x{{k}}x{{d}}xf64 : memref<{{n}}x{{k}}x{{d}}xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_{{k}}xf64 : memref<{{k}}xf64> = dense<0.000000e+00>
  func @enzyme_gmm_objective_full(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>, %arg2: memref<{{k}}x{{d}}xf64>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64>, %arg4: memref<{{n}}x{{d}}xf64>, %arg5: f64, %arg6: i64) -> f64 {
    %0 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%0 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = math.exp %arg7 : f64
      linalg.yield %39 : f64
    }
    %1 = memref.get_global @__constant_{{k}}xf64 : memref<{{k}}xf64>
    %2 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%1, %2) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%2 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = arith.addf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }
    %3 = memref.get_global @__constant_{{n}}x{{k}}x{{d}}xf64 : memref<{{n}}x{{k}}x{{d}}xf64>
    %4 = memref.alloc() : memref<{{n}}x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1 : memref<{{n}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) outs(%4 : memref<{{n}}x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %39 = arith.subf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }
    %5 = memref.get_global @__constant_{{n}}x{{k}}x{{d}}xf64 : memref<{{n}}x{{k}}x{{d}}xf64>
    %6 = memref.alloc() : memref<{{n}}x{{k}}x{{d}}xf64>
    linalg.copy(%5, %6) : memref<{{n}}x{{k}}x{{d}}xf64>, memref<{{n}}x{{k}}x{{d}}xf64> 
    linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg3, %4 : memref<{{k}}x{{d}}x{{d}}xf64>, memref<{{n}}x{{k}}x{{d}}xf64>) outs(%6 : memref<{{n}}x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %39 = arith.mulf %arg7, %arg8 : f64
      %40 = arith.addf %39, %arg9 : f64
      linalg.yield %40 : f64
    }
    %7 = memref.get_global @__constant_{{n}}x{{k}}x{{d}}xf64 : memref<{{n}}x{{k}}x{{d}}xf64>
    %8 = memref.alloc() : memref<{{n}}x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %4, %6 : memref<{{k}}x{{d}}xf64>, memref<{{n}}x{{k}}x{{d}}xf64>, memref<{{n}}x{{k}}x{{d}}xf64>) outs(%8 : memref<{{n}}x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %39 = arith.mulf %arg7, %arg8 : f64
      %40 = arith.addf %39, %arg9 : f64
      %41 = arith.mulf %40, %40 : f64
      linalg.yield %41 : f64
    }
    memref.dealloc %6 : memref<{{n}}x{{k}}x{{d}}xf64>
    memref.dealloc %4 : memref<{{n}}x{{k}}x{{d}}xf64>
    %9 = memref.get_global @__constant_{{n}}x{{k}}xf64 : memref<{{n}}x{{k}}xf64>
    %10 = memref.alloc() : memref<{{n}}x{{k}}xf64>
    linalg.copy(%9, %10) : memref<{{n}}x{{k}}xf64>, memref<{{n}}x{{k}}xf64> 
    linalg.generic {indexing_maps = [#map4, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%8 : memref<{{n}}x{{k}}x{{d}}xf64>) outs(%10 : memref<{{n}}x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = arith.addf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }
    memref.dealloc %8 : memref<{{n}}x{{k}}x{{d}}xf64>
    %11 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%arg0, %2 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%11 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %39 = arith.addf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }
    %12 = memref.get_global @__constant_{{n}}x{{k}}xf64 : memref<{{n}}x{{k}}xf64>
    %cst = arith.constant 5.000000e-01 : f64
    %13 = memref.alloc() : memref<{{n}}x{{k}}xf64>
    linalg.generic {indexing_maps = [#map10, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%11, %10 : memref<{{k}}xf64>, memref<{{n}}x{{k}}xf64>) outs(%13 : memref<{{n}}x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %39 = arith.mulf %arg8, %cst : f64
      %40 = arith.subf %arg7, %39 : f64
      linalg.yield %40 : f64
    }
    memref.dealloc %11 : memref<{{k}}xf64>
    memref.dealloc %10 : memref<{{n}}x{{k}}xf64>
    %14 = memref.get_global @__constant_{{n}}xf64 : memref<{{n}}xf64>
    %15 = memref.alloc() : memref<{{n}}xf64>
    linalg.copy(%14, %15) : memref<{{n}}xf64>, memref<{{n}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%13 : memref<{{n}}x{{k}}xf64>) outs(%15 : memref<{{n}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = arith.cmpf ogt, %arg7, %arg8 : f64
      %40 = scf.if %39 -> (f64) {
        scf.yield %arg7 : f64
      } else {
        scf.yield %arg8 : f64
      }
      linalg.yield %40 : f64
    }
    %16 = memref.get_global @__constant_{{n}}xf64_0 : memref<{{n}}xf64>
    %17 = memref.alloc() : memref<{{n}}xf64>
    linalg.copy(%16, %17) : memref<{{n}}xf64>, memref<{{n}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%13, %15 : memref<{{n}}x{{k}}xf64>, memref<{{n}}xf64>) outs(%17 : memref<{{n}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %39 = arith.subf %arg7, %arg8 : f64
      %40 = math.exp %39 : f64
      %41 = arith.addf %40, %arg9 : f64
      linalg.yield %41 : f64
    }
    memref.dealloc %13 : memref<{{n}}x{{k}}xf64>
    %18 = memref.alloc() : memref<{{n}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel"]} ins(%17 : memref<{{n}}xf64>) outs(%18 : memref<{{n}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = math.log %arg7 : f64
      linalg.yield %39 : f64
    }
    memref.dealloc %17 : memref<{{n}}xf64>
    %19 = memref.alloc() : memref<{{n}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%18, %15 : memref<{{n}}xf64>, memref<{{n}}xf64>) outs(%19 : memref<{{n}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %39 = arith.addf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }
    memref.dealloc %18 : memref<{{n}}xf64>
    memref.dealloc %15 : memref<{{n}}xf64>
    %20 = memref.get_global @__constant_xf64 : memref<f64>
    %21 = memref.alloc() : memref<f64>
    linalg.copy(%20, %21) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%19 : memref<{{n}}xf64>) outs(%21 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = arith.addf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }
    memref.dealloc %19 : memref<{{n}}xf64>
    %22 = memref.get_global @__constant_{{k}}xf64 : memref<{{k}}xf64>
    %23 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%22, %23) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%0 : memref<{{k}}x{{d}}xf64>) outs(%23 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = arith.mulf %arg7, %arg7 : f64
      %40 = arith.addf %39, %arg8 : f64
      linalg.yield %40 : f64
    }
    memref.dealloc %0 : memref<{{k}}x{{d}}xf64>
    %24 = memref.get_global @__constant_{{k}}xf64 : memref<{{k}}xf64>
    %25 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%24, %25) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map4, #map12], iterator_types = ["parallel", "reduction", "reduction"]} ins(%arg3 : memref<{{k}}x{{d}}x{{d}}xf64>) outs(%25 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = arith.mulf %arg7, %arg7 : f64
      %40 = arith.addf %39, %arg8 : f64
      linalg.yield %40 : f64
    }
    %26 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%23, %25 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%26 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %39 = arith.addf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }
    memref.dealloc %25 : memref<{{k}}xf64>
    memref.dealloc %23 : memref<{{k}}xf64>
    %27 = memref.get_global @__constant_xf64 : memref<f64>
    %28 = memref.alloc() : memref<f64>
    linalg.copy(%27, %28) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%26 : memref<{{k}}xf64>) outs(%28 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = arith.mulf %cst, %arg5 : f64
      %40 = arith.mulf %39, %arg5 : f64
      %41 = arith.mulf %40, %arg7 : f64
      %42 = arith.addf %41, %arg8 : f64
      linalg.yield %42 : f64
    }
    memref.dealloc %26 : memref<{{k}}xf64>
    %29 = memref.get_global @__constant_xf64 : memref<f64>
    %30 = memref.alloc() : memref<f64>
    linalg.copy(%29, %30) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%2 : memref<{{k}}xf64>) outs(%30 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = arith.sitofp %arg6 : i64 to f64
      %40 = arith.mulf %39, %arg7 : f64
      %41 = arith.addf %40, %arg8 : f64
      linalg.yield %41 : f64
    }
    memref.dealloc %2 : memref<{{k}}xf64>
    %31 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%28, %30 : memref<f64>, memref<f64>) outs(%31 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %39 = arith.subf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }
    memref.dealloc %30 : memref<f64>
    memref.dealloc %28 : memref<f64>
    %32 = memref.get_global @__constant_xf64 : memref<f64>
    %33 = memref.alloc() : memref<f64>
    linalg.copy(%32, %33) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%arg0 : memref<{{k}}xf64>) outs(%33 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = math.exp %arg7 : f64
      %40 = arith.addf %39, %arg8 : f64
      linalg.yield %40 : f64
    }
    %34 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%33 : memref<f64>) outs(%34 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %39 = math.log %arg7 : f64
      linalg.yield %39 : f64
    }
    memref.dealloc %33 : memref<f64>
    %35 = memref.get_global @__constant_xf64_0 : memref<f64>
    %36 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%35, %34 : memref<f64>, memref<f64>) outs(%36 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %39 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }
    memref.dealloc %34 : memref<f64>
    %37 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%21, %36 : memref<f64>, memref<f64>) outs(%37 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %39 = arith.subf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }
    memref.dealloc %36 : memref<f64>
    memref.dealloc %21 : memref<f64>
    %38 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%37, %31 : memref<f64>, memref<f64>) outs(%38 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %39 = arith.addf %arg7, %arg8 : f64
      linalg.yield %39 : f64
    }
    memref.dealloc %37 : memref<f64>
    memref.dealloc %31 : memref<f64>

    %final_val = memref.load %38[] : memref<f64>
    return %final_val : f64
  }

  func @enzyme_gmm_full(
    %arg0: memref<{{k}}xf64>,
    %arg1: memref<{{k}}x{{d}}xf64>,
    %arg2: memref<{{k}}x{{d}}xf64>,
    %arg3: memref<{{k}}x{{d}}x{{d}}xf64>,
    %arg4: memref<{{n}}x{{d}}xf64>,
    %arg5: f64,
    %arg6: i64
  ) -> (
    memref<{{k}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}x{{d}}xf64>
  ) {
    %darg0 = memref.alloc() : memref<{{k}}xf64>
    %darg1 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    %darg2 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    %darg3 = memref.alloc() : memref<{{k}}x{{d}}x{{d}}xf64>

    %f = constant @enzyme_gmm_objective_full : (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>, f64, i64) -> f64
    %df = standalone.diff %f {const = [4]} : (
      memref<{{k}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}x{{d}}xf64>,
      memref<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> f64, (
      memref<{{k}}xf64>,
      memref<{{k}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}x{{d}}xf64>,
      memref<{{k}}x{{d}}x{{d}}xf64>,
      memref<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> f64
    call_indirect %df(%arg0, %darg0, %arg1, %darg1, %arg2, %darg2, %arg3, %darg3, %arg4, %arg5, %arg6) : (
      memref<{{k}}xf64>,
      memref<{{k}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}x{{d}}xf64>,
      memref<{{k}}x{{d}}x{{d}}xf64>,
      memref<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> f64
    return %darg0, %darg1, %darg2, %darg3 : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>
  }
}
