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
  memref.global "private" constant @__constant_{{k}}x{{d}}xf64 : memref<{{k}}x{{d}}xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_{{k}}x{{d}}x{{d}}xf64 : memref<{{k}}x{{d}}x{{d}}xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf64_1 : memref<f64> = dense<1.000000e+00>
  memref.global "private" constant @__constant_xf64_0 : memref<f64> = dense<1.000000e+03>
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_{{k}}xf64 : memref<{{k}}xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1000xf64_0 : memref<1000xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1000xf64 : memref<1000xf64> = dense<-1.000000e+09>
  memref.global "private" constant @__constant_1000x{{k}}xf64 : memref<1000x{{k}}xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1000x{{k}}x{{d}}xf64 : memref<1000x{{k}}x{{d}}xf64> = dense<0.000000e+00>
  func @lagrad_gmm_objective_full(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>, %arg2: memref<{{k}}x{{d}}xf64>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64>, %arg4: memref<1000x{{d}}xf64>, %arg5: f64, %arg6: i64) -> memref<f64> {
    %cst = arith.constant 5.000000e-01 : f64
    %0 = memref.get_global @__constant_1000x{{k}}x{{d}}xf64 : memref<1000x{{k}}x{{d}}xf64>
    %1 = memref.get_global @__constant_1000x{{k}}xf64 : memref<1000x{{k}}xf64>
    %2 = memref.get_global @__constant_1000xf64 : memref<1000xf64>
    %3 = memref.get_global @__constant_1000xf64_0 : memref<1000xf64>
    %4 = memref.get_global @__constant_{{k}}xf64 : memref<{{k}}xf64>
    %5 = memref.get_global @__constant_xf64 : memref<f64>
    %6 = memref.get_global @__constant_xf64_0 : memref<f64>
    %7 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%7 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %31 = math.exp %arg7 : f64
      linalg.yield %31 : f64
    }
    %8 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%4, %8) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%8 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %31 = arith.addf %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    %9 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1 : memref<1000x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) outs(%9 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %31 = arith.subf %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    %10 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.copy(%0, %10) : memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64> 
    linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg3, %9 : memref<{{k}}x{{d}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%10 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %31 = arith.mulf %arg7, %arg8 : f64
      %32 = arith.addf %31, %arg9 : f64
      linalg.yield %32 : f64
    }
    %11 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %9, %10 : memref<{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%11 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %31 = arith.mulf %arg7, %arg8 : f64
      %32 = arith.addf %31, %arg9 : f64
      %33 = arith.mulf %32, %32 : f64
      linalg.yield %33 : f64
    }
    memref.dealloc %10 : memref<1000x{{k}}x{{d}}xf64>
    memref.dealloc %9 : memref<1000x{{k}}x{{d}}xf64>
    %12 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.copy(%1, %12) : memref<1000x{{k}}xf64>, memref<1000x{{k}}xf64> 
    linalg.generic {indexing_maps = [#map4, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11 : memref<1000x{{k}}x{{d}}xf64>) outs(%12 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %31 = arith.addf %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    memref.dealloc %11 : memref<1000x{{k}}x{{d}}xf64>
    %13 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%arg0, %8 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%13 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %31 = arith.addf %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    %14 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.generic {indexing_maps = [#map10, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%13, %12 : memref<{{k}}xf64>, memref<1000x{{k}}xf64>) outs(%14 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %31 = arith.mulf %arg8, %cst : f64
      %32 = arith.subf %arg7, %31 : f64
      linalg.yield %32 : f64
    }
    memref.dealloc %13 : memref<{{k}}xf64>
    memref.dealloc %12 : memref<1000x{{k}}xf64>
    %15 = memref.alloc() : memref<1000xf64>
    linalg.copy(%2, %15) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%14 : memref<1000x{{k}}xf64>) outs(%15 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %31 = arith.cmpf ogt, %arg7, %arg8 : f64
      %32 = select %31, %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    %16 = memref.alloc() : memref<1000xf64>
    linalg.copy(%3, %16) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%14, %15 : memref<1000x{{k}}xf64>, memref<1000xf64>) outs(%16 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %31 = arith.subf %arg7, %arg8 : f64
      %32 = math.exp %31 : f64
      %33 = arith.addf %32, %arg9 : f64
      linalg.yield %33 : f64
    }
    memref.dealloc %14 : memref<1000x{{k}}xf64>
    %17 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel"]} ins(%16 : memref<1000xf64>) outs(%17 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %31 = math.log %arg7 : f64
      linalg.yield %31 : f64
    }
    memref.dealloc %16 : memref<1000xf64>
    %18 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%17, %15 : memref<1000xf64>, memref<1000xf64>) outs(%18 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %31 = arith.addf %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    memref.dealloc %17 : memref<1000xf64>
    memref.dealloc %15 : memref<1000xf64>
    %19 = memref.alloc() : memref<f64>
    linalg.copy(%5, %19) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%18 : memref<1000xf64>) outs(%19 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %31 = arith.addf %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    memref.dealloc %18 : memref<1000xf64>
    %20 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%4, %20) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%7 : memref<{{k}}x{{d}}xf64>) outs(%20 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %31 = arith.mulf %arg7, %arg7 : f64
      %32 = arith.addf %31, %arg8 : f64
      linalg.yield %32 : f64
    }
    memref.dealloc %7 : memref<{{k}}x{{d}}xf64>
    %21 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%4, %21) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map4, #map12], iterator_types = ["parallel", "reduction", "reduction"]} ins(%arg3 : memref<{{k}}x{{d}}x{{d}}xf64>) outs(%21 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %31 = arith.mulf %arg7, %arg7 : f64
      %32 = arith.addf %31, %arg8 : f64
      linalg.yield %32 : f64
    }
    %22 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%20, %21 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%22 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %31 = arith.addf %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    memref.dealloc %21 : memref<{{k}}xf64>
    memref.dealloc %20 : memref<{{k}}xf64>
    %23 = memref.alloc() : memref<f64>
    linalg.copy(%5, %23) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%22 : memref<{{k}}xf64>) outs(%23 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %31 = arith.mulf %cst, %arg5 : f64
      %32 = arith.mulf %31, %arg5 : f64
      %33 = arith.mulf %32, %arg7 : f64
      %34 = arith.addf %33, %arg8 : f64
      linalg.yield %34 : f64
    }
    memref.dealloc %22 : memref<{{k}}xf64>
    %24 = memref.alloc() : memref<f64>
    linalg.copy(%5, %24) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%8 : memref<{{k}}xf64>) outs(%24 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %31 = arith.sitofp %arg6 : i64 to f64
      %32 = arith.mulf %31, %arg7 : f64
      %33 = arith.addf %32, %arg8 : f64
      linalg.yield %33 : f64
    }
    memref.dealloc %8 : memref<{{k}}xf64>
    %25 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%23, %24 : memref<f64>, memref<f64>) outs(%25 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %31 = arith.subf %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    memref.dealloc %24 : memref<f64>
    memref.dealloc %23 : memref<f64>
    %26 = memref.alloc() : memref<f64>
    linalg.copy(%5, %26) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%arg0 : memref<{{k}}xf64>) outs(%26 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %31 = math.exp %arg7 : f64
      %32 = arith.addf %31, %arg8 : f64
      linalg.yield %32 : f64
    }
    %27 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%26 : memref<f64>) outs(%27 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %31 = math.log %arg7 : f64
      linalg.yield %31 : f64
    }
    memref.dealloc %26 : memref<f64>
    %28 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%6, %27 : memref<f64>, memref<f64>) outs(%28 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %31 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    memref.dealloc %27 : memref<f64>
    %29 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%19, %28 : memref<f64>, memref<f64>) outs(%29 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %31 = arith.subf %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    memref.dealloc %28 : memref<f64>
    memref.dealloc %19 : memref<f64>
    %30 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%29, %25 : memref<f64>, memref<f64>) outs(%30 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %31 = arith.addf %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    memref.dealloc %29 : memref<f64>
    memref.dealloc %25 : memref<f64>
    return %30 : memref<f64>
  }
  func @__grad_lagrad_gmm_objective_full(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>, %arg2: memref<{{k}}x{{d}}xf64>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64>, %arg4: memref<1000x{{d}}xf64>, %arg5: f64, %arg6: i64) -> (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e-01 : f64
    %0 = memref.get_global @__constant_1000xf64 : memref<1000xf64>
    %1 = memref.get_global @__constant_xf64_0 : memref<f64>
    %2 = memref.get_global @__constant_xf64_1 : memref<f64>
    %3 = memref.get_global @__constant_xf64 : memref<f64>
    %4 = memref.get_global @__constant_1000xf64_0 : memref<1000xf64>
    %5 = memref.get_global @__constant_{{k}}xf64 : memref<{{k}}xf64>
    %6 = memref.get_global @__constant_1000x{{k}}xf64 : memref<1000x{{k}}xf64>
    %7 = memref.get_global @__constant_{{k}}x{{d}}x{{d}}xf64 : memref<{{k}}x{{d}}x{{d}}xf64>
    %8 = memref.get_global @__constant_1000x{{k}}x{{d}}xf64 : memref<1000x{{k}}x{{d}}xf64>
    %9 = memref.get_global @__constant_{{k}}x{{d}}xf64 : memref<{{k}}x{{d}}xf64>
    %10 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%10 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %59 = math.exp %arg7 : f64
      linalg.yield %59 : f64
    }
    %11 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%5, %11) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%11 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    %12 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1 : memref<1000x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) outs(%12 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.subf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    %13 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.copy(%8, %13) : memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64> 
    linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg3, %12 : memref<{{k}}x{{d}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%13 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.mulf %arg7, %arg8 : f64
      %60 = arith.addf %59, %arg9 : f64
      linalg.yield %60 : f64
    }
    %14 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %12, %13 : memref<{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%14 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %59 = arith.mulf %arg7, %arg8 : f64
      %60 = arith.addf %59, %arg9 : f64
      %61 = arith.mulf %60, %60 : f64
      linalg.yield %61 : f64
    }
    %15 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.copy(%6, %15) : memref<1000x{{k}}xf64>, memref<1000x{{k}}xf64> 
    linalg.generic {indexing_maps = [#map4, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%14 : memref<1000x{{k}}x{{d}}xf64>) outs(%15 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    %16 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%arg0, %11 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%16 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    %17 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.generic {indexing_maps = [#map10, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%16, %15 : memref<{{k}}xf64>, memref<1000x{{k}}xf64>) outs(%17 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.mulf %arg8, %cst_0 : f64
      %60 = arith.subf %arg7, %59 : f64
      linalg.yield %60 : f64
    }
    %18 = memref.alloc() : memref<1000xf64>
    linalg.copy(%0, %18) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%17 : memref<1000x{{k}}xf64>) outs(%18 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %59 = arith.cmpf ogt, %arg7, %arg8 : f64
      %60 = select %59, %arg7, %arg8 : f64
      linalg.yield %60 : f64
    }
    %19 = memref.alloc() : memref<1000xf64>
    linalg.copy(%4, %19) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%17, %18 : memref<1000x{{k}}xf64>, memref<1000xf64>) outs(%19 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.subf %arg7, %arg8 : f64
      %60 = math.exp %59 : f64
      %61 = arith.addf %60, %arg9 : f64
      linalg.yield %61 : f64
    }
    %20 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel"]} ins(%19 : memref<1000xf64>) outs(%20 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %59 = math.log %arg7 : f64
      linalg.yield %59 : f64
    }
    %21 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%20, %18 : memref<1000xf64>, memref<1000xf64>) outs(%21 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %20 : memref<1000xf64>
    %22 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%5, %22) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%10 : memref<{{k}}x{{d}}xf64>) outs(%22 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %59 = arith.mulf %arg7, %arg7 : f64
      %60 = arith.addf %59, %arg8 : f64
      linalg.yield %60 : f64
    }
    %23 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%5, %23) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map4, #map12], iterator_types = ["parallel", "reduction", "reduction"]} ins(%arg3 : memref<{{k}}x{{d}}x{{d}}xf64>) outs(%23 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %59 = arith.mulf %arg7, %arg7 : f64
      %60 = arith.addf %59, %arg8 : f64
      linalg.yield %60 : f64
    }
    %24 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%22, %23 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%24 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %23 : memref<{{k}}xf64>
    memref.dealloc %22 : memref<{{k}}xf64>
    %25 = memref.alloc() : memref<f64>
    linalg.copy(%3, %25) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%arg0 : memref<{{k}}xf64>) outs(%25 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %59 = math.exp %arg7 : f64
      %60 = arith.addf %59, %arg8 : f64
      linalg.yield %60 : f64
    }
    %26 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%2 : memref<f64>) outs(%26 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %59 = arith.negf %arg7 : f64
      linalg.yield %59 : f64
    }
    %27 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%26, %1 : memref<f64>, memref<f64>) outs(%27 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %26 : memref<f64>
    %28 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%27, %25 : memref<f64>, memref<f64>) outs(%28 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.divf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %27 : memref<f64>
    memref.dealloc %25 : memref<f64>
    %29 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%arg0, %28 : memref<{{k}}xf64>, memref<f64>) outs(%29 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = math.exp %arg7 : f64
      %60 = arith.mulf %arg8, %59 : f64
      linalg.yield %60 : f64
    }
    memref.dealloc %28 : memref<f64>
    %30 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%2 : memref<f64>) outs(%30 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %59 = arith.negf %arg7 : f64
      linalg.yield %59 : f64
    }
    %31 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%11, %30 : memref<{{k}}xf64>, memref<f64>) outs(%31 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.sitofp %arg6 : i64 to f64
      %60 = arith.mulf %arg8, %59 : f64
      linalg.yield %60 : f64
    }
    memref.dealloc %30 : memref<f64>
    memref.dealloc %11 : memref<{{k}}xf64>
    %32 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%24, %2 : memref<{{k}}xf64>, memref<f64>) outs(%32 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.mulf %cst_0, %arg5 : f64
      %60 = arith.mulf %59, %arg5 : f64
      %61 = arith.mulf %arg8, %60 : f64
      linalg.yield %61 : f64
    }
    memref.dealloc %24 : memref<{{k}}xf64>
    %33 = memref.alloc() : memref<{{k}}x{{d}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map4, #map12, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg3, %32 : memref<{{k}}x{{d}}x{{d}}xf64>, memref<{{k}}xf64>) outs(%33 : memref<{{k}}x{{d}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.mulf %arg8, %arg7 : f64
      %60 = arith.mulf %arg8, %arg7 : f64
      %61 = arith.addf %59, %60 : f64
      linalg.yield %61 : f64
    }
    %34 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%10, %32 : memref<{{k}}x{{d}}xf64>, memref<{{k}}xf64>) outs(%34 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.mulf %arg8, %arg7 : f64
      %60 = arith.mulf %arg8, %arg7 : f64
      %61 = arith.addf %59, %60 : f64
      linalg.yield %61 : f64
    }
    memref.dealloc %32 : memref<{{k}}xf64>
    %35 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%21, %2 : memref<1000xf64>, memref<f64>) outs(%35 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      linalg.yield %arg8 : f64
    }
    memref.dealloc %21 : memref<1000xf64>
    %36 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%35, %19 : memref<1000xf64>, memref<1000xf64>) outs(%36 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.divf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %19 : memref<1000xf64>
    %37 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.generic {indexing_maps = [#map0, #map1, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%17, %18, %36 : memref<1000x{{k}}xf64>, memref<1000xf64>, memref<1000xf64>) outs(%37 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %59 = arith.subf %arg7, %arg8 : f64
      %60 = math.exp %59 : f64
      %61 = arith.mulf %arg9, %60 : f64
      linalg.yield %61 : f64
    }
    %38 = memref.alloc() : memref<1000xf64>
    linalg.copy(%4, %38) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%17, %18, %36 : memref<1000x{{k}}xf64>, memref<1000xf64>, memref<1000xf64>) outs(%38 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %59 = arith.subf %arg7, %arg8 : f64
      %60 = math.exp %59 : f64
      %61 = arith.mulf %arg9, %60 : f64
      %62 = arith.negf %61 : f64
      %63 = arith.addf %62, %arg10 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %36 : memref<1000xf64>
    memref.dealloc %18 : memref<1000xf64>
    %39 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%35, %38 : memref<1000xf64>, memref<1000xf64>) outs(%39 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %38 : memref<1000xf64>
    memref.dealloc %35 : memref<1000xf64>
    %40 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.copy(%6, %40) : memref<1000x{{k}}xf64>, memref<1000x{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%17, %39 : memref<1000x{{k}}xf64>, memref<1000xf64>) outs(%40 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.cmpf ogt, %arg7, %arg9 : f64
      %60 = select %59, %arg8, %cst : f64
      linalg.yield %60 : f64
    }
    memref.dealloc %39 : memref<1000xf64>
    memref.dealloc %17 : memref<1000x{{k}}xf64>
    %41 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%37, %40 : memref<1000x{{k}}xf64>, memref<1000x{{k}}xf64>) outs(%41 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %40 : memref<1000x{{k}}xf64>
    memref.dealloc %37 : memref<1000x{{k}}xf64>
    %42 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%5, %42) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map10, #map0, #map0, #map10], iterator_types = ["parallel", "parallel"]} ins(%16, %15, %41 : memref<{{k}}xf64>, memref<1000x{{k}}xf64>, memref<1000x{{k}}xf64>) outs(%42 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %59 = arith.addf %arg9, %arg10 : f64
      linalg.yield %59 : f64
    }
    %43 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.generic {indexing_maps = [#map10, #map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%16, %15, %41 : memref<{{k}}xf64>, memref<1000x{{k}}xf64>, memref<1000x{{k}}xf64>) outs(%43 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %59 = arith.negf %arg9 : f64
      %60 = arith.mulf %59, %cst_0 : f64
      linalg.yield %60 : f64
    }
    memref.dealloc %41 : memref<1000x{{k}}xf64>
    memref.dealloc %16 : memref<{{k}}xf64>
    memref.dealloc %15 : memref<1000x{{k}}xf64>
    %44 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%29, %42 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%44 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %29 : memref<{{k}}xf64>
    %45 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%31, %42 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%45 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %42 : memref<{{k}}xf64>
    memref.dealloc %31 : memref<{{k}}xf64>
    %46 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map4, #map8, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14, %43 : memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}xf64>) outs(%46 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      linalg.yield %arg8 : f64
    }
    memref.dealloc %43 : memref<1000x{{k}}xf64>
    memref.dealloc %14 : memref<1000x{{k}}x{{d}}xf64>
    %47 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.copy(%9, %47) : memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64> 
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %12, %13, %46 : memref<{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%47 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
      %59 = arith.mulf %arg7, %arg8 : f64
      %60 = arith.addf %59, %arg9 : f64
      %61 = arith.mulf %arg10, %60 : f64
      %62 = arith.mulf %arg10, %60 : f64
      %63 = arith.addf %61, %62 : f64
      %64 = arith.mulf %63, %arg8 : f64
      %65 = arith.addf %64, %arg11 : f64
      linalg.yield %65 : f64
    }
    %48 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%34, %47 : memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) outs(%48 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %47 : memref<{{k}}x{{d}}xf64>
    memref.dealloc %34 : memref<{{k}}x{{d}}xf64>
    %49 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %12, %13, %46 : memref<{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%49 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
      %59 = arith.mulf %arg7, %arg8 : f64
      %60 = arith.addf %59, %arg9 : f64
      %61 = arith.mulf %arg10, %60 : f64
      %62 = arith.mulf %arg10, %60 : f64
      %63 = arith.addf %61, %62 : f64
      %64 = arith.mulf %63, %arg7 : f64
      linalg.yield %64 : f64
    }
    %50 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %12, %13, %46 : memref<{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%50 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
      %59 = arith.mulf %arg7, %arg8 : f64
      %60 = arith.addf %59, %arg9 : f64
      %61 = arith.mulf %arg10, %60 : f64
      %62 = arith.mulf %arg10, %60 : f64
      %63 = arith.addf %61, %62 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %46 : memref<1000x{{k}}x{{d}}xf64>
    memref.dealloc %13 : memref<1000x{{k}}x{{d}}xf64>
    %51 = memref.alloc() : memref<{{k}}x{{d}}x{{d}}xf64>
    linalg.copy(%7, %51) : memref<{{k}}x{{d}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64> 
    linalg.generic {indexing_maps = [#map5, #map6, #map7, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3, %12, %50 : memref<{{k}}x{{d}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%51 : memref<{{k}}x{{d}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %59 = arith.mulf %arg9, %arg8 : f64
      %60 = arith.addf %59, %arg10 : f64
      linalg.yield %60 : f64
    }
    %52 = memref.alloc() : memref<{{k}}x{{d}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %51 : memref<{{k}}x{{d}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>) outs(%52 : memref<{{k}}x{{d}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %51 : memref<{{k}}x{{d}}x{{d}}xf64>
    memref.dealloc %33 : memref<{{k}}x{{d}}x{{d}}xf64>
    %53 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.copy(%8, %53) : memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64> 
    linalg.generic {indexing_maps = [#map5, #map6, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3, %12, %50 : memref<{{k}}x{{d}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%53 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %59 = arith.mulf %arg9, %arg7 : f64
      %60 = arith.addf %59, %arg10 : f64
      linalg.yield %60 : f64
    }
    memref.dealloc %50 : memref<1000x{{k}}x{{d}}xf64>
    memref.dealloc %12 : memref<1000x{{k}}x{{d}}xf64>
    %54 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %53 : memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%54 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %53 : memref<1000x{{k}}x{{d}}xf64>
    memref.dealloc %49 : memref<1000x{{k}}x{{d}}xf64>
    %55 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.copy(%9, %55) : memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64> 
    linalg.generic {indexing_maps = [#map2, #map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1, %54 : memref<1000x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%55 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %59 = arith.negf %arg9 : f64
      %60 = arith.addf %59, %arg10 : f64
      linalg.yield %60 : f64
    }
    memref.dealloc %54 : memref<1000x{{k}}x{{d}}xf64>
    %56 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %45 : memref<{{k}}x{{d}}xf64>, memref<{{k}}xf64>) outs(%56 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      linalg.yield %arg8 : f64
    }
    memref.dealloc %45 : memref<{{k}}xf64>
    %57 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%48, %10 : memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) outs(%57 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %48 : memref<{{k}}x{{d}}xf64>
    memref.dealloc %10 : memref<{{k}}x{{d}}xf64>
    %58 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%56, %57 : memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) outs(%58 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %59 = arith.addf %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    memref.dealloc %57 : memref<{{k}}x{{d}}xf64>
    memref.dealloc %56 : memref<{{k}}x{{d}}xf64>
    return %44, %55, %58, %52 : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>
  }
  func @lagrad_gmm_objective_tri(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>, %arg2: memref<{{k}}x{{d}}xf64>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64>, %arg4: memref<1000x{{d}}xf64>, %arg5: f64, %arg6: i64) -> memref<f64> {
    %cst = arith.constant 5.000000e-01 : f64
    %c1000 = arith.constant 1000 : index
    %ck = arith.constant {{k}} : index
    %cd = arith.constant {{d}} : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = memref.get_global @__constant_1000x{{k}}xf64 : memref<1000x{{k}}xf64>
    %1 = memref.get_global @__constant_1000xf64 : memref<1000xf64>
    %2 = memref.get_global @__constant_1000xf64_0 : memref<1000xf64>
    %3 = memref.get_global @__constant_{{k}}xf64 : memref<{{k}}xf64>
    %4 = memref.get_global @__constant_xf64 : memref<f64>
    %5 = memref.get_global @__constant_xf64_0 : memref<f64>
    %6 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%6 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = math.exp %arg7 : f64
      linalg.yield %32 : f64
    }
    %7 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%3, %7) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%7 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = arith.addf %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    %8 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1 : memref<1000x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) outs(%8 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %32 = arith.subf %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    %9 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.fill(%cst_0, %9) : f64, memref<1000x{{k}}x{{d}}xf64> 
    %10 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %c0) -> (index) {
      %32 = scf.for %arg9 = %c0 to %ck step %c1 iter_args(%arg10 = %arg8) -> (index) {
        %33 = scf.for %arg11 = %c0 to %cd step %c1 iter_args(%arg12 = %arg10) -> (index) {
          %34 = scf.for %arg13 = %c0 to %arg11 step %c1 iter_args(%arg14 = %arg12) -> (index) {
            %35 = memref.load %arg3[%arg9, %arg11, %arg13] : memref<{{k}}x{{d}}x{{d}}xf64>
            %36 = memref.load %8[%arg7, %arg9, %arg13] : memref<1000x{{k}}x{{d}}xf64>
            %37 = memref.load %9[%arg7, %arg9, %arg11] : memref<1000x{{k}}x{{d}}xf64>
            %38 = arith.mulf %35, %36 : f64
            %39 = arith.addf %38, %37 : f64
            memref.store %39, %9[%arg7, %arg9, %arg11] : memref<1000x{{k}}x{{d}}xf64>
            %40 = arith.addi %arg14, %c1 : index
            scf.yield %40 : index
          }
          scf.yield %34 : index
        }
        scf.yield %33 : index
      }
      scf.yield %32 : index
    }
    %11 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %8, %9 : memref<{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%11 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %32 = arith.mulf %arg7, %arg8 : f64
      %33 = arith.addf %32, %arg9 : f64
      %34 = arith.mulf %33, %33 : f64
      linalg.yield %34 : f64
    }
    memref.dealloc %9 : memref<1000x{{k}}x{{d}}xf64>
    memref.dealloc %8 : memref<1000x{{k}}x{{d}}xf64>
    %12 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.copy(%0, %12) : memref<1000x{{k}}xf64>, memref<1000x{{k}}xf64> 
    linalg.generic {indexing_maps = [#map4, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11 : memref<1000x{{k}}x{{d}}xf64>) outs(%12 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = arith.addf %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    memref.dealloc %11 : memref<1000x{{k}}x{{d}}xf64>
    %13 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%arg0, %7 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%13 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %32 = arith.addf %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    %14 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.generic {indexing_maps = [#map10, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%13, %12 : memref<{{k}}xf64>, memref<1000x{{k}}xf64>) outs(%14 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %32 = arith.mulf %arg8, %cst : f64
      %33 = arith.subf %arg7, %32 : f64
      linalg.yield %33 : f64
    }
    memref.dealloc %13 : memref<{{k}}xf64>
    memref.dealloc %12 : memref<1000x{{k}}xf64>
    %15 = memref.alloc() : memref<1000xf64>
    linalg.copy(%1, %15) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%14 : memref<1000x{{k}}xf64>) outs(%15 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = arith.cmpf ogt, %arg7, %arg8 : f64
      %33 = select %32, %arg7, %arg8 : f64
      linalg.yield %33 : f64
    }
    %16 = memref.alloc() : memref<1000xf64>
    linalg.copy(%2, %16) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%14, %15 : memref<1000x{{k}}xf64>, memref<1000xf64>) outs(%16 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %32 = arith.subf %arg7, %arg8 : f64
      %33 = math.exp %32 : f64
      %34 = arith.addf %33, %arg9 : f64
      linalg.yield %34 : f64
    }
    memref.dealloc %14 : memref<1000x{{k}}xf64>
    %17 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel"]} ins(%16 : memref<1000xf64>) outs(%17 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = math.log %arg7 : f64
      linalg.yield %32 : f64
    }
    memref.dealloc %16 : memref<1000xf64>
    %18 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%17, %15 : memref<1000xf64>, memref<1000xf64>) outs(%18 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %32 = arith.addf %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    memref.dealloc %17 : memref<1000xf64>
    memref.dealloc %15 : memref<1000xf64>
    %19 = memref.alloc() : memref<f64>
    linalg.copy(%4, %19) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%18 : memref<1000xf64>) outs(%19 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = arith.addf %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    memref.dealloc %18 : memref<1000xf64>
    %20 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%3, %20) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%6 : memref<{{k}}x{{d}}xf64>) outs(%20 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = arith.mulf %arg7, %arg7 : f64
      %33 = arith.addf %32, %arg8 : f64
      linalg.yield %33 : f64
    }
    memref.dealloc %6 : memref<{{k}}x{{d}}xf64>
    %21 = memref.alloc() : memref<{{k}}xf64>
    linalg.fill(%cst_0, %21) : f64, memref<{{k}}xf64> 
    %22 = scf.for %arg7 = %c0 to %ck step %c1 iter_args(%arg8 = %c0) -> (index) {
      %32 = scf.for %arg9 = %c0 to %cd step %c1 iter_args(%arg10 = %arg8) -> (index) {
        %33 = scf.for %arg11 = %c0 to %arg9 step %c1 iter_args(%arg12 = %arg10) -> (index) {
          %34 = memref.load %arg3[%arg7, %arg9, %arg11] : memref<{{k}}x{{d}}x{{d}}xf64>
          %35 = memref.load %21[%arg7] : memref<{{k}}xf64>
          %36 = arith.mulf %34, %34 : f64
          %37 = arith.addf %36, %35 : f64
          memref.store %37, %21[%arg7] : memref<{{k}}xf64>
          %38 = arith.addi %arg12, %c1 : index
          scf.yield %38 : index
        }
        scf.yield %33 : index
      }
      scf.yield %32 : index
    }
    %23 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%20, %21 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%23 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %32 = arith.addf %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    memref.dealloc %21 : memref<{{k}}xf64>
    memref.dealloc %20 : memref<{{k}}xf64>
    %24 = memref.alloc() : memref<f64>
    linalg.copy(%4, %24) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%23 : memref<{{k}}xf64>) outs(%24 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = arith.mulf %cst, %arg5 : f64
      %33 = arith.mulf %32, %arg5 : f64
      %34 = arith.mulf %33, %arg7 : f64
      %35 = arith.addf %34, %arg8 : f64
      linalg.yield %35 : f64
    }
    memref.dealloc %23 : memref<{{k}}xf64>
    %25 = memref.alloc() : memref<f64>
    linalg.copy(%4, %25) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%7 : memref<{{k}}xf64>) outs(%25 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = arith.sitofp %arg6 : i64 to f64
      %33 = arith.mulf %32, %arg7 : f64
      %34 = arith.addf %33, %arg8 : f64
      linalg.yield %34 : f64
    }
    memref.dealloc %7 : memref<{{k}}xf64>
    %26 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%24, %25 : memref<f64>, memref<f64>) outs(%26 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %32 = arith.subf %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    memref.dealloc %25 : memref<f64>
    memref.dealloc %24 : memref<f64>
    %27 = memref.alloc() : memref<f64>
    linalg.copy(%4, %27) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%arg0 : memref<{{k}}xf64>) outs(%27 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = math.exp %arg7 : f64
      %33 = arith.addf %32, %arg8 : f64
      linalg.yield %33 : f64
    }
    %28 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%27 : memref<f64>) outs(%28 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = math.log %arg7 : f64
      linalg.yield %32 : f64
    }
    memref.dealloc %27 : memref<f64>
    %29 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%5, %28 : memref<f64>, memref<f64>) outs(%29 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %32 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    memref.dealloc %28 : memref<f64>
    %30 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%19, %29 : memref<f64>, memref<f64>) outs(%30 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %32 = arith.subf %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    memref.dealloc %29 : memref<f64>
    memref.dealloc %19 : memref<f64>
    %31 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%30, %26 : memref<f64>, memref<f64>) outs(%31 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %32 = arith.addf %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    memref.dealloc %30 : memref<f64>
    memref.dealloc %26 : memref<f64>
    return %31 : memref<f64>
  }
  func @__grad_lagrad_gmm_objective_tri(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>, %arg2: memref<{{k}}x{{d}}xf64>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64>, %arg4: memref<1000x{{d}}xf64>, %arg5: f64, %arg6: i64) -> (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>) {
    %cst = arith.constant 5.000000e-01 : f64
    %ck = arith.constant {{k}} : index
    %cd = arith.constant {{d}} : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = memref.get_global @__constant_1000xf64 : memref<1000xf64>
    %1 = memref.get_global @__constant_xf64_0 : memref<f64>
    %2 = memref.get_global @__constant_xf64_1 : memref<f64>
    %3 = memref.get_global @__constant_xf64 : memref<f64>
    %4 = memref.get_global @__constant_1000xf64_0 : memref<1000xf64>
    %5 = memref.get_global @__constant_{{k}}xf64 : memref<{{k}}xf64>
    %6 = memref.get_global @__constant_1000x{{k}}xf64 : memref<1000x{{k}}xf64>
    %7 = memref.get_global @__constant_{{k}}x{{d}}xf64 : memref<{{k}}x{{d}}xf64>
    %8 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%8 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %63 = math.exp %arg7 : f64
      linalg.yield %63 : f64
    }
    %9 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%5, %9) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%9 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %63 = arith.addf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    %10 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1 : memref<1000x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) outs(%10 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.subf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    %11 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.fill(%cst_0, %11) : f64, memref<1000x{{k}}x{{d}}xf64> 
    %12 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %c0) -> (index) {
      %63 = scf.for %arg9 = %c0 to %ck step %c1 iter_args(%arg10 = %arg8) -> (index) {
        %64 = scf.for %arg11 = %c0 to %cd step %c1 iter_args(%arg12 = %arg10) -> (index) {
          %65 = scf.for %arg13 = %c0 to %arg11 step %c1 iter_args(%arg14 = %arg12) -> (index) {
            %66 = memref.load %arg3[%arg9, %arg11, %arg13] : memref<{{k}}x{{d}}x{{d}}xf64>
            %67 = memref.load %10[%arg7, %arg9, %arg13] : memref<1000x{{k}}x{{d}}xf64>
            %68 = memref.load %11[%arg7, %arg9, %arg11] : memref<1000x{{k}}x{{d}}xf64>
            %69 = arith.mulf %66, %67 : f64
            %70 = arith.addf %69, %68 : f64
            memref.store %70, %11[%arg7, %arg9, %arg11] : memref<1000x{{k}}x{{d}}xf64>
            %71 = arith.addi %arg14, %c1 : index
            scf.yield %71 : index
          }
          scf.yield %65 : index
        }
        scf.yield %64 : index
      }
      scf.yield %63 : index
    }
    %13 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %10, %11 : memref<{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%13 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %63 = arith.mulf %arg7, %arg8 : f64
      %64 = arith.addf %63, %arg9 : f64
      %65 = arith.mulf %64, %64 : f64
      linalg.yield %65 : f64
    }
    %14 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.copy(%6, %14) : memref<1000x{{k}}xf64>, memref<1000x{{k}}xf64> 
    linalg.generic {indexing_maps = [#map4, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13 : memref<1000x{{k}}x{{d}}xf64>) outs(%14 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %63 = arith.addf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    %15 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%arg0, %9 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%15 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.addf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    %16 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.generic {indexing_maps = [#map10, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%15, %14 : memref<{{k}}xf64>, memref<1000x{{k}}xf64>) outs(%16 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.mulf %arg8, %cst : f64
      %64 = arith.subf %arg7, %63 : f64
      linalg.yield %64 : f64
    }
    %17 = memref.alloc() : memref<1000xf64>
    linalg.copy(%0, %17) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%16 : memref<1000x{{k}}xf64>) outs(%17 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %63 = arith.cmpf ogt, %arg7, %arg8 : f64
      %64 = select %63, %arg7, %arg8 : f64
      linalg.yield %64 : f64
    }
    %18 = memref.alloc() : memref<1000xf64>
    linalg.copy(%4, %18) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%16, %17 : memref<1000x{{k}}xf64>, memref<1000xf64>) outs(%18 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.subf %arg7, %arg8 : f64
      %64 = math.exp %63 : f64
      %65 = arith.addf %64, %arg9 : f64
      linalg.yield %65 : f64
    }
    %19 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel"]} ins(%18 : memref<1000xf64>) outs(%19 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %63 = math.log %arg7 : f64
      linalg.yield %63 : f64
    }
    %20 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%19, %17 : memref<1000xf64>, memref<1000xf64>) outs(%20 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.addf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %19 : memref<1000xf64>
    %21 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%5, %21) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%8 : memref<{{k}}x{{d}}xf64>) outs(%21 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %63 = arith.mulf %arg7, %arg7 : f64
      %64 = arith.addf %63, %arg8 : f64
      linalg.yield %64 : f64
    }
    %22 = memref.alloc() : memref<{{k}}xf64>
    linalg.fill(%cst_0, %22) : f64, memref<{{k}}xf64> 
    %23 = scf.for %arg7 = %c0 to %ck step %c1 iter_args(%arg8 = %c0) -> (index) {
      %63 = scf.for %arg9 = %c0 to %cd step %c1 iter_args(%arg10 = %arg8) -> (index) {
        %64 = scf.for %arg11 = %c0 to %arg9 step %c1 iter_args(%arg12 = %arg10) -> (index) {
          %65 = memref.load %arg3[%arg7, %arg9, %arg11] : memref<{{k}}x{{d}}x{{d}}xf64>
          %66 = memref.load %22[%arg7] : memref<{{k}}xf64>
          %67 = arith.mulf %65, %65 : f64
          %68 = arith.addf %67, %66 : f64
          memref.store %68, %22[%arg7] : memref<{{k}}xf64>
          %69 = arith.addi %arg12, %c1 : index
          scf.yield %69 : index
        }
        scf.yield %64 : index
      }
      scf.yield %63 : index
    }
    %24 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%21, %22 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%24 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.addf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %22 : memref<{{k}}xf64>
    memref.dealloc %21 : memref<{{k}}xf64>
    %25 = memref.alloc() : memref<f64>
    linalg.copy(%3, %25) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%arg0 : memref<{{k}}xf64>) outs(%25 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %63 = math.exp %arg7 : f64
      %64 = arith.addf %63, %arg8 : f64
      linalg.yield %64 : f64
    }
    %26 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%2 : memref<f64>) outs(%26 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %63 = arith.negf %arg7 : f64
      linalg.yield %63 : f64
    }
    %27 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%26, %1 : memref<f64>, memref<f64>) outs(%27 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %26 : memref<f64>
    %28 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%27, %25 : memref<f64>, memref<f64>) outs(%28 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.divf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %27 : memref<f64>
    memref.dealloc %25 : memref<f64>
    %29 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%arg0, %28 : memref<{{k}}xf64>, memref<f64>) outs(%29 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = math.exp %arg7 : f64
      %64 = arith.mulf %arg8, %63 : f64
      linalg.yield %64 : f64
    }
    memref.dealloc %28 : memref<f64>
    %30 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%2 : memref<f64>) outs(%30 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %63 = arith.negf %arg7 : f64
      linalg.yield %63 : f64
    }
    %31 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%9, %30 : memref<{{k}}xf64>, memref<f64>) outs(%31 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.sitofp %arg6 : i64 to f64
      %64 = arith.mulf %arg8, %63 : f64
      linalg.yield %64 : f64
    }
    memref.dealloc %30 : memref<f64>
    memref.dealloc %9 : memref<{{k}}xf64>
    %32 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%24, %2 : memref<{{k}}xf64>, memref<f64>) outs(%32 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.mulf %cst, %arg5 : f64
      %64 = arith.mulf %63, %arg5 : f64
      %65 = arith.mulf %arg8, %64 : f64
      linalg.yield %65 : f64
    }
    memref.dealloc %24 : memref<{{k}}xf64>
    %33 = memref.alloc() : memref<{{k}}x{{d}}x{{d}}xf64>
    %34 = scf.for %arg7 = %c0 to %ck step %c1 iter_args(%arg8 = %c0) -> (index) {
      %63 = scf.for %arg9 = %c0 to %cd step %c1 iter_args(%arg10 = %arg8) -> (index) {
        %64 = scf.for %arg11 = %c0 to %arg9 step %c1 iter_args(%arg12 = %arg10) -> (index) {
          %65 = memref.load %arg3[%arg7, %arg9, %arg11] : memref<{{k}}x{{d}}x{{d}}xf64>
          %66 = memref.load %32[%arg7] : memref<{{k}}xf64>
          %67 = arith.mulf %66, %65 : f64
          %68 = arith.mulf %66, %65 : f64
          %69 = arith.addf %67, %68 : f64
          memref.store %69, %33[%arg7, %arg9, %arg11] : memref<{{k}}x{{d}}x{{d}}xf64>
          %70 = arith.addi %arg12, %c1 : index
          scf.yield %70 : index
        }
        scf.yield %64 : index
      }
      scf.yield %63 : index
    }
    %35 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%8, %32 : memref<{{k}}x{{d}}xf64>, memref<{{k}}xf64>) outs(%35 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.mulf %arg8, %arg7 : f64
      %64 = arith.mulf %arg8, %arg7 : f64
      %65 = arith.addf %63, %64 : f64
      linalg.yield %65 : f64
    }
    memref.dealloc %32 : memref<{{k}}xf64>
    %36 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%20, %2 : memref<1000xf64>, memref<f64>) outs(%36 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      linalg.yield %arg8 : f64
    }
    memref.dealloc %20 : memref<1000xf64>
    %37 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%36, %18 : memref<1000xf64>, memref<1000xf64>) outs(%37 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.divf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %18 : memref<1000xf64>
    %38 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.generic {indexing_maps = [#map0, #map1, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%16, %17, %37 : memref<1000x{{k}}xf64>, memref<1000xf64>, memref<1000xf64>) outs(%38 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %63 = arith.subf %arg7, %arg8 : f64
      %64 = math.exp %63 : f64
      %65 = arith.mulf %arg9, %64 : f64
      linalg.yield %65 : f64
    }
    %39 = memref.alloc() : memref<1000xf64>
    linalg.copy(%4, %39) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%16, %17, %37 : memref<1000x{{k}}xf64>, memref<1000xf64>, memref<1000xf64>) outs(%39 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %63 = arith.subf %arg7, %arg8 : f64
      %64 = math.exp %63 : f64
      %65 = arith.mulf %arg9, %64 : f64
      %66 = arith.negf %65 : f64
      %67 = arith.addf %66, %arg10 : f64
      linalg.yield %67 : f64
    }
    memref.dealloc %37 : memref<1000xf64>
    memref.dealloc %17 : memref<1000xf64>
    %40 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%36, %39 : memref<1000xf64>, memref<1000xf64>) outs(%40 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.addf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %39 : memref<1000xf64>
    memref.dealloc %36 : memref<1000xf64>
    %41 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.copy(%6, %41) : memref<1000x{{k}}xf64>, memref<1000x{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%16, %40 : memref<1000x{{k}}xf64>, memref<1000xf64>) outs(%41 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.cmpf ogt, %arg7, %arg9 : f64
      %64 = select %63, %arg8, %cst_0 : f64
      linalg.yield %64 : f64
    }
    memref.dealloc %40 : memref<1000xf64>
    memref.dealloc %16 : memref<1000x{{k}}xf64>
    %42 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%38, %41 : memref<1000x{{k}}xf64>, memref<1000x{{k}}xf64>) outs(%42 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.addf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %41 : memref<1000x{{k}}xf64>
    memref.dealloc %38 : memref<1000x{{k}}xf64>
    %43 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%5, %43) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map10, #map0, #map0, #map10], iterator_types = ["parallel", "parallel"]} ins(%15, %14, %42 : memref<{{k}}xf64>, memref<1000x{{k}}xf64>, memref<1000x{{k}}xf64>) outs(%43 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %63 = arith.addf %arg9, %arg10 : f64
      linalg.yield %63 : f64
    }
    %44 = memref.alloc() : memref<1000x{{k}}xf64>
    linalg.generic {indexing_maps = [#map10, #map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%15, %14, %42 : memref<{{k}}xf64>, memref<1000x{{k}}xf64>, memref<1000x{{k}}xf64>) outs(%44 : memref<1000x{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %63 = arith.negf %arg9 : f64
      %64 = arith.mulf %63, %cst : f64
      linalg.yield %64 : f64
    }
    memref.dealloc %42 : memref<1000x{{k}}xf64>
    memref.dealloc %15 : memref<{{k}}xf64>
    memref.dealloc %14 : memref<1000x{{k}}xf64>
    %45 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%29, %43 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%45 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.addf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %29 : memref<{{k}}xf64>
    %46 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%31, %43 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%46 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.addf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %43 : memref<{{k}}xf64>
    memref.dealloc %31 : memref<{{k}}xf64>
    %47 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map4, #map8, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%13, %44 : memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}xf64>) outs(%47 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      linalg.yield %arg8 : f64
    }
    memref.dealloc %44 : memref<1000x{{k}}xf64>
    memref.dealloc %13 : memref<1000x{{k}}x{{d}}xf64>
    %48 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.copy(%7, %48) : memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64> 
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %10, %11, %47 : memref<{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%48 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
      %63 = arith.mulf %arg7, %arg8 : f64
      %64 = arith.addf %63, %arg9 : f64
      %65 = arith.mulf %arg10, %64 : f64
      %66 = arith.mulf %arg10, %64 : f64
      %67 = arith.addf %65, %66 : f64
      %68 = arith.mulf %67, %arg8 : f64
      %69 = arith.addf %68, %arg11 : f64
      linalg.yield %69 : f64
    }
    %49 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%35, %48 : memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) outs(%49 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.addf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %48 : memref<{{k}}x{{d}}xf64>
    memref.dealloc %35 : memref<{{k}}x{{d}}xf64>
    %50 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %10, %11, %47 : memref<{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%50 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
      %63 = arith.mulf %arg7, %arg8 : f64
      %64 = arith.addf %63, %arg9 : f64
      %65 = arith.mulf %arg10, %64 : f64
      %66 = arith.mulf %arg10, %64 : f64
      %67 = arith.addf %65, %66 : f64
      %68 = arith.mulf %67, %arg7 : f64
      linalg.yield %68 : f64
    }
    %51 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %10, %11, %47 : memref<{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%51 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
      %63 = arith.mulf %arg7, %arg8 : f64
      %64 = arith.addf %63, %arg9 : f64
      %65 = arith.mulf %arg10, %64 : f64
      %66 = arith.mulf %arg10, %64 : f64
      %67 = arith.addf %65, %66 : f64
      linalg.yield %67 : f64
    }
    memref.dealloc %47 : memref<1000x{{k}}x{{d}}xf64>
    memref.dealloc %11 : memref<1000x{{k}}x{{d}}xf64>
    %52 = memref.alloc() : memref<{{k}}x{{d}}x{{d}}xf64>
    linalg.fill(%cst_0, %52) : f64, memref<{{k}}x{{d}}x{{d}}xf64> 
    %53 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %c0) -> (index) {
      %63 = scf.for %arg9 = %c0 to %ck step %c1 iter_args(%arg10 = %arg8) -> (index) {
        %64 = scf.for %arg11 = %c0 to %cd step %c1 iter_args(%arg12 = %arg10) -> (index) {
          %65 = scf.for %arg13 = %c0 to %arg11 step %c1 iter_args(%arg14 = %arg12) -> (index) {
            %66 = memref.load %10[%arg7, %arg9, %arg13] : memref<1000x{{k}}x{{d}}xf64>
            %67 = memref.load %51[%arg7, %arg9, %arg11] : memref<1000x{{k}}x{{d}}xf64>
            %68 = memref.load %52[%arg9, %arg11, %arg13] : memref<{{k}}x{{d}}x{{d}}xf64>
            %69 = arith.mulf %67, %66 : f64
            %70 = arith.addf %69, %68 : f64
            memref.store %70, %52[%arg9, %arg11, %arg13] : memref<{{k}}x{{d}}x{{d}}xf64>
            %71 = arith.addi %arg14, %c1 : index
            scf.yield %71 : index
          }
          scf.yield %65 : index
        }
        scf.yield %64 : index
      }
      scf.yield %63 : index
    }
    memref.dealloc %10 : memref<1000x{{k}}x{{d}}xf64>
    %54 = memref.alloc() : memref<{{k}}x{{d}}x{{d}}xf64>
    %55 = scf.for %arg7 = %c0 to %ck step %c1 iter_args(%arg8 = %c0) -> (index) {
      %63 = scf.for %arg9 = %c0 to %cd step %c1 iter_args(%arg10 = %arg8) -> (index) {
        %64 = scf.for %arg11 = %c0 to %arg9 step %c1 iter_args(%arg12 = %arg10) -> (index) {
          %65 = memref.load %33[%arg7, %arg9, %arg11] : memref<{{k}}x{{d}}x{{d}}xf64>
          %66 = memref.load %52[%arg7, %arg9, %arg11] : memref<{{k}}x{{d}}x{{d}}xf64>
          %67 = arith.addf %65, %66 : f64
          memref.store %67, %54[%arg7, %arg9, %arg11] : memref<{{k}}x{{d}}x{{d}}xf64>
          %68 = arith.addi %arg12, %c1 : index
          scf.yield %68 : index
        }
        scf.yield %64 : index
      }
      scf.yield %63 : index
    }
    memref.dealloc %52 : memref<{{k}}x{{d}}x{{d}}xf64>
    memref.dealloc %33 : memref<{{k}}x{{d}}x{{d}}xf64>
    %56 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.fill(%cst_0, %56) : f64, memref<1000x{{k}}x{{d}}xf64> 
    %57 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %c0) -> (index) {
      %63 = scf.for %arg9 = %c0 to %ck step %c1 iter_args(%arg10 = %arg8) -> (index) {
        %64 = scf.for %arg11 = %c0 to %cd step %c1 iter_args(%arg12 = %arg10) -> (index) {
          %65 = scf.for %arg13 = %c0 to %arg11 step %c1 iter_args(%arg14 = %arg12) -> (index) {
            %66 = memref.load %arg3[%arg9, %arg11, %arg13] : memref<{{k}}x{{d}}x{{d}}xf64>
            %67 = memref.load %51[%arg7, %arg9, %arg11] : memref<1000x{{k}}x{{d}}xf64>
            %68 = memref.load %56[%arg7, %arg9, %arg13] : memref<1000x{{k}}x{{d}}xf64>
            %69 = arith.mulf %67, %66 : f64
            %70 = arith.addf %69, %68 : f64
            memref.store %70, %56[%arg7, %arg9, %arg13] : memref<1000x{{k}}x{{d}}xf64>
            %71 = arith.addi %arg14, %c1 : index
            scf.yield %71 : index
          }
          scf.yield %65 : index
        }
        scf.yield %64 : index
      }
      scf.yield %63 : index
    }
    memref.dealloc %51 : memref<1000x{{k}}x{{d}}xf64>
    %58 = memref.alloc() : memref<1000x{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %56 : memref<1000x{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%58 : memref<1000x{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.addf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %56 : memref<1000x{{k}}x{{d}}xf64>
    memref.dealloc %50 : memref<1000x{{k}}x{{d}}xf64>
    %59 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.copy(%7, %59) : memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64> 
    linalg.generic {indexing_maps = [#map2, #map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1, %58 : memref<1000x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<1000x{{k}}x{{d}}xf64>) outs(%59 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %63 = arith.negf %arg9 : f64
      %64 = arith.addf %63, %arg10 : f64
      linalg.yield %64 : f64
    }
    memref.dealloc %58 : memref<1000x{{k}}x{{d}}xf64>
    %60 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %46 : memref<{{k}}x{{d}}xf64>, memref<{{k}}xf64>) outs(%60 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      linalg.yield %arg8 : f64
    }
    memref.dealloc %46 : memref<{{k}}xf64>
    %61 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%49, %8 : memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) outs(%61 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %49 : memref<{{k}}x{{d}}xf64>
    memref.dealloc %8 : memref<{{k}}x{{d}}xf64>
    %62 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%60, %61 : memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) outs(%62 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %63 = arith.addf %arg7, %arg8 : f64
      linalg.yield %63 : f64
    }
    memref.dealloc %61 : memref<{{k}}x{{d}}xf64>
    memref.dealloc %60 : memref<{{k}}x{{d}}xf64>
    return %45, %59, %62, %54 : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>
  }
  func @lagrad_gmm_full(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>, %arg2: memref<{{k}}x{{d}}xf64>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64>, %arg4: memref<1000x{{d}}xf64>, %arg5: f64, %arg6: i64) -> (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>) {
    %0:4 = call @__grad_lagrad_gmm_objective_full(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>, memref<1000x{{d}}xf64>, f64, i64) -> (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>)
    return %0#0, %0#1, %0#2, %0#3 : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>
  }
  func @lagrad_gmm_tri(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>, %arg2: memref<{{k}}x{{d}}xf64>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64>, %arg4: memref<1000x{{d}}xf64>, %arg5: f64, %arg6: i64) -> (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>) {
    %0:4 = call @__grad_lagrad_gmm_objective_tri(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>, memref<1000x{{d}}xf64>, f64, i64) -> (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>)
    return %0#0, %0#1, %0#2, %0#3 : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>
  }
}

