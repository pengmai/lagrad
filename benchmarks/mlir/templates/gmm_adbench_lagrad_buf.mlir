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
  memref.global "private" constant @__constant_200x128x128xf64 : memref<200x128x128xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_200x128xf64 : memref<200x128xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1000x200x128xf64 : memref<1000x200x128xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1000x200xf64 : memref<1000x200xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1000xf64 : memref<1000xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_200xf64 : memref<200xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf64_0 : memref<f64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<1.000000e+03>
  func private @print_memref_f64(memref<*xf64>) attributes {llvm.emit_c_interface}
  func @lagrad_gmm_objective_full(%arg0: memref<200xf64>, %arg1: memref<200x128xf64>, %arg2: memref<200x128xf64>, %arg3: memref<200x128x128xf64>, %arg4: memref<1000x128xf64>, %arg5: f64, %arg6: i64) -> memref<f64> {
    %cst = arith.constant 5.000000e-01 : f64
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %1 = memref.get_global @__constant_xf64_0 : memref<f64>
    %2 = memref.get_global @__constant_200xf64 : memref<200xf64>
    %3 = memref.get_global @__constant_1000xf64 : memref<1000xf64>
    %4 = memref.get_global @__constant_1000x200xf64 : memref<1000x200xf64>
    %5 = memref.get_global @__constant_1000x200x128xf64 : memref<1000x200x128xf64>
    %6 = memref.alloc() : memref<200x128xf64>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<200x128xf64>) outs(%6 : memref<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %30 = math.exp %arg7 : f64
      linalg.yield %30 : f64
    }
    %7 = memref.alloc() : memref<200xf64>
    linalg.copy(%2, %7) : memref<200xf64>, memref<200xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<200x128xf64>) outs(%7 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %30 = arith.addf %arg7, %arg8 : f64
      linalg.yield %30 : f64
    }
    %8 = memref.alloc() : memref<1000x200x128xf64>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1 : memref<1000x128xf64>, memref<200x128xf64>) outs(%8 : memref<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %30 = arith.subf %arg7, %arg8 : f64
      linalg.yield %30 : f64
    }
    %9 = memref.alloc() : memref<1000x200x128xf64>
    linalg.copy(%5, %9) : memref<1000x200x128xf64>, memref<1000x200x128xf64> 
    linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg3, %8 : memref<200x128x128xf64>, memref<1000x200x128xf64>) outs(%9 : memref<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %30 = arith.mulf %arg7, %arg8 : f64
      %31 = arith.addf %30, %arg9 : f64
      linalg.yield %31 : f64
    }
    %10 = memref.alloc() : memref<1000x200x128xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %8, %9 : memref<200x128xf64>, memref<1000x200x128xf64>, memref<1000x200x128xf64>) outs(%10 : memref<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %30 = arith.mulf %arg7, %arg8 : f64
      %31 = arith.addf %30, %arg9 : f64
      %32 = arith.mulf %31, %31 : f64
      linalg.yield %32 : f64
    }
    %11 = memref.alloc() : memref<1000x200xf64>
    linalg.copy(%4, %11) : memref<1000x200xf64>, memref<1000x200xf64> 
    linalg.generic {indexing_maps = [#map4, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10 : memref<1000x200x128xf64>) outs(%11 : memref<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %30 = arith.addf %arg7, %arg8 : f64
      linalg.yield %30 : f64
    }
    %12 = memref.alloc() : memref<200xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%arg0, %7 : memref<200xf64>, memref<200xf64>) outs(%12 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %30 = arith.addf %arg7, %arg8 : f64
      linalg.yield %30 : f64
    }
    %13 = memref.alloc() : memref<1000x200xf64>
    linalg.generic {indexing_maps = [#map10, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%12, %11 : memref<200xf64>, memref<1000x200xf64>) outs(%13 : memref<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %30 = arith.mulf %arg8, %cst : f64
      %31 = arith.subf %arg7, %30 : f64
      linalg.yield %31 : f64
    }
    %14 = memref.alloc() : memref<1000xf64>
    linalg.copy(%3, %14) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%13 : memref<1000x200xf64>) outs(%14 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %30 = arith.cmpf ogt, %arg7, %arg8 : f64
      %31 = select %30, %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    %15 = memref.alloc() : memref<1000xf64>
    linalg.copy(%3, %15) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%13, %14 : memref<1000x200xf64>, memref<1000xf64>) outs(%15 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %30 = arith.subf %arg7, %arg8 : f64
      %31 = math.exp %30 : f64
      %32 = arith.addf %31, %arg9 : f64
      linalg.yield %32 : f64
    }
    %16 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel"]} ins(%15 : memref<1000xf64>) outs(%16 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %30 = math.log %arg7 : f64
      linalg.yield %30 : f64
    }
    %17 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%16, %14 : memref<1000xf64>, memref<1000xf64>) outs(%17 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %30 = arith.addf %arg7, %arg8 : f64
      linalg.yield %30 : f64
    }
    %18 = memref.alloca() : memref<f64>
    linalg.copy(%1, %18) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%17 : memref<1000xf64>) outs(%18 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %30 = arith.addf %arg7, %arg8 : f64
      linalg.yield %30 : f64
    }
    %19 = memref.alloc() : memref<200xf64>
    linalg.copy(%2, %19) : memref<200xf64>, memref<200xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%6 : memref<200x128xf64>) outs(%19 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %30 = arith.mulf %arg7, %arg7 : f64
      %31 = arith.addf %30, %arg8 : f64
      linalg.yield %31 : f64
    }
    %20 = memref.alloc() : memref<200xf64>
    linalg.copy(%2, %20) : memref<200xf64>, memref<200xf64> 
    linalg.generic {indexing_maps = [#map4, #map12], iterator_types = ["parallel", "reduction", "reduction"]} ins(%arg3 : memref<200x128x128xf64>) outs(%20 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %30 = arith.mulf %arg7, %arg7 : f64
      %31 = arith.addf %30, %arg8 : f64
      linalg.yield %31 : f64
    }
    %21 = memref.alloc() : memref<200xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%19, %20 : memref<200xf64>, memref<200xf64>) outs(%21 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %30 = arith.addf %arg7, %arg8 : f64
      linalg.yield %30 : f64
    }
    %22 = memref.alloca() : memref<f64>
    linalg.copy(%1, %22) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%21 : memref<200xf64>) outs(%22 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %30 = arith.mulf %cst, %arg5 : f64
      %31 = arith.mulf %30, %arg5 : f64
      %32 = arith.mulf %31, %arg7 : f64
      %33 = arith.addf %32, %arg8 : f64
      linalg.yield %33 : f64
    }
    %23 = memref.alloca() : memref<f64>
    linalg.copy(%1, %23) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%7 : memref<200xf64>) outs(%23 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %30 = arith.sitofp %arg6 : i64 to f64
      %31 = arith.mulf %30, %arg7 : f64
      %32 = arith.addf %31, %arg8 : f64
      linalg.yield %32 : f64
    }
    %24 = memref.alloca() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%22, %23 : memref<f64>, memref<f64>) outs(%24 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %30 = arith.subf %arg7, %arg8 : f64
      linalg.yield %30 : f64
    }
    %25 = memref.alloca() : memref<f64>
    linalg.copy(%1, %25) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%arg0 : memref<200xf64>) outs(%25 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %30 = math.exp %arg7 : f64
      %31 = arith.addf %30, %arg8 : f64
      linalg.yield %31 : f64
    }
    %26 = memref.alloca() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%25 : memref<f64>) outs(%26 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %30 = math.log %arg7 : f64
      linalg.yield %30 : f64
    }
    %27 = memref.alloca() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%0, %26 : memref<f64>, memref<f64>) outs(%27 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %30 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %30 : f64
    }
    %28 = memref.alloca() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%18, %27 : memref<f64>, memref<f64>) outs(%28 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %30 = arith.subf %arg7, %arg8 : f64
      linalg.yield %30 : f64
    }
    %29 = memref.alloc() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%28, %24 : memref<f64>, memref<f64>) outs(%29 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %30 = arith.addf %arg7, %arg8 : f64
      linalg.yield %30 : f64
    }
    return %29 : memref<f64>
  }
  func @__grad_lagrad_gmm_objective_full(%arg0: memref<200xf64>, %arg1: memref<200x128xf64>, %arg2: memref<200x128xf64>, %arg3: memref<200x128x128xf64>, %arg4: memref<1000x128xf64>, %arg5: f64, %arg6: i64) -> (memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x128x128xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 5.000000e-01 : f64
    %0 = memref.get_global @__constant_200x128xf64 : memref<200x128xf64>
    %1 = memref.get_global @__constant_1000x200x128xf64 : memref<1000x200x128xf64>
    %2 = memref.get_global @__constant_200x128x128xf64 : memref<200x128x128xf64>
    %3 = memref.get_global @__constant_1000x200xf64 : memref<1000x200xf64>
    %4 = memref.get_global @__constant_200xf64 : memref<200xf64>
    %5 = memref.get_global @__constant_xf64_0 : memref<f64>
    %6 = memref.get_global @__constant_1000xf64 : memref<1000xf64>
    %7 = memref.get_global @__constant_xf64 : memref<f64>
    %8 = memref.alloc() : memref<200x128xf64>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<200x128xf64>) outs(%8 : memref<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %58 = math.exp %arg7 : f64
      linalg.yield %58 : f64
    }
    %9 = memref.alloc() : memref<200xf64>
    linalg.copy(%4, %9) : memref<200xf64>, memref<200xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<200x128xf64>) outs(%9 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %10 = memref.alloc() : memref<1000x200x128xf64>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1 : memref<1000x128xf64>, memref<200x128xf64>) outs(%10 : memref<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.subf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %11 = memref.alloc() : memref<1000x200x128xf64>
    linalg.copy(%1, %11) : memref<1000x200x128xf64>, memref<1000x200x128xf64> 
    linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg3, %10 : memref<200x128x128xf64>, memref<1000x200x128xf64>) outs(%11 : memref<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.mulf %arg7, %arg8 : f64
      %59 = arith.addf %58, %arg9 : f64
      linalg.yield %59 : f64
    }
    %12 = memref.alloc() : memref<1000x200x128xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %10, %11 : memref<200x128xf64>, memref<1000x200x128xf64>, memref<1000x200x128xf64>) outs(%12 : memref<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %58 = arith.mulf %arg7, %arg8 : f64
      %59 = arith.addf %58, %arg9 : f64
      %60 = arith.mulf %59, %59 : f64
      linalg.yield %60 : f64
    }
    %13 = memref.alloc() : memref<1000x200xf64>
    linalg.copy(%3, %13) : memref<1000x200xf64>, memref<1000x200xf64> 
    linalg.generic {indexing_maps = [#map4, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%12 : memref<1000x200x128xf64>) outs(%13 : memref<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %14 = memref.alloc() : memref<200xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%arg0, %9 : memref<200xf64>, memref<200xf64>) outs(%14 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %15 = memref.alloc() : memref<1000x200xf64>
    linalg.generic {indexing_maps = [#map10, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%14, %13 : memref<200xf64>, memref<1000x200xf64>) outs(%15 : memref<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.mulf %arg8, %cst_1 : f64
      %59 = arith.subf %arg7, %58 : f64
      linalg.yield %59 : f64
    }
    %16 = memref.alloc() : memref<1000xf64>
    linalg.copy(%6, %16) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%15 : memref<1000x200xf64>) outs(%16 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %58 = arith.cmpf ogt, %arg7, %arg8 : f64
      %59 = select %58, %arg7, %arg8 : f64
      linalg.yield %59 : f64
    }
    %17 = memref.alloc() : memref<1000xf64>
    linalg.copy(%6, %17) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%15, %16 : memref<1000x200xf64>, memref<1000xf64>) outs(%17 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.subf %arg7, %arg8 : f64
      %59 = math.exp %58 : f64
      %60 = arith.addf %59, %arg9 : f64
      linalg.yield %60 : f64
    }
    %18 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel"]} ins(%17 : memref<1000xf64>) outs(%18 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %58 = math.log %arg7 : f64
      linalg.yield %58 : f64
    }
    %19 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%18, %16 : memref<1000xf64>, memref<1000xf64>) outs(%19 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %20 = memref.alloc() : memref<200xf64>
    linalg.copy(%4, %20) : memref<200xf64>, memref<200xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%8 : memref<200x128xf64>) outs(%20 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %58 = arith.mulf %arg7, %arg7 : f64
      %59 = arith.addf %58, %arg8 : f64
      linalg.yield %59 : f64
    }
    %21 = memref.alloc() : memref<200xf64>
    linalg.copy(%4, %21) : memref<200xf64>, memref<200xf64> 
    linalg.generic {indexing_maps = [#map4, #map12], iterator_types = ["parallel", "reduction", "reduction"]} ins(%arg3 : memref<200x128x128xf64>) outs(%21 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %58 = arith.mulf %arg7, %arg7 : f64
      %59 = arith.addf %58, %arg8 : f64
      linalg.yield %59 : f64
    }
    %22 = memref.alloc() : memref<200xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%20, %21 : memref<200xf64>, memref<200xf64>) outs(%22 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %23 = memref.alloca() : memref<f64>
    linalg.copy(%5, %23) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%arg0 : memref<200xf64>) outs(%23 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %58 = math.exp %arg7 : f64
      %59 = arith.addf %58, %arg8 : f64
      linalg.yield %59 : f64
    }
    %24 = memref.alloca() : memref<f64>
    linalg.fill(%cst_0, %24) : f64, memref<f64> 
    %25 = memref.alloca() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%24 : memref<f64>) outs(%25 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %58 = arith.negf %arg7 : f64
      linalg.yield %58 : f64
    }
    %26 = memref.alloca() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%25, %7 : memref<f64>, memref<f64>) outs(%26 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %27 = memref.alloca() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%26, %23 : memref<f64>, memref<f64>) outs(%27 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.divf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %28 = memref.alloc() : memref<200xf64>
    linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%arg0, %27 : memref<200xf64>, memref<f64>) outs(%28 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = math.exp %arg7 : f64
      %59 = arith.mulf %arg8, %58 : f64
      linalg.yield %59 : f64
    }
    %29 = memref.alloca() : memref<f64>
    linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%24 : memref<f64>) outs(%29 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %58 = arith.negf %arg7 : f64
      linalg.yield %58 : f64
    }
    %30 = memref.alloc() : memref<200xf64>
    linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%9, %29 : memref<200xf64>, memref<f64>) outs(%30 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.sitofp %arg6 : i64 to f64
      %59 = arith.mulf %arg8, %58 : f64
      linalg.yield %59 : f64
    }
    %31 = memref.alloc() : memref<200xf64>
    linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%22, %24 : memref<200xf64>, memref<f64>) outs(%31 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.mulf %cst_1, %arg5 : f64
      %59 = arith.mulf %58, %arg5 : f64
      %60 = arith.mulf %arg8, %59 : f64
      linalg.yield %60 : f64
    }
    %32 = memref.alloc() : memref<200x128x128xf64>
    linalg.generic {indexing_maps = [#map4, #map12, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg3, %31 : memref<200x128x128xf64>, memref<200xf64>) outs(%32 : memref<200x128x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.mulf %arg8, %arg7 : f64
      %59 = arith.mulf %arg8, %arg7 : f64
      %60 = arith.addf %58, %59 : f64
      linalg.yield %60 : f64
    }
    %33 = memref.alloc() : memref<200x128xf64>
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%8, %31 : memref<200x128xf64>, memref<200xf64>) outs(%33 : memref<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.mulf %arg8, %arg7 : f64
      %59 = arith.mulf %arg8, %arg7 : f64
      %60 = arith.addf %58, %59 : f64
      linalg.yield %60 : f64
    }
    %34 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%19, %24 : memref<1000xf64>, memref<f64>) outs(%34 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      linalg.yield %arg8 : f64
    }
    %35 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%34, %17 : memref<1000xf64>, memref<1000xf64>) outs(%35 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.divf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %36 = memref.alloc() : memref<1000x200xf64>
    linalg.generic {indexing_maps = [#map0, #map1, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%15, %16, %35 : memref<1000x200xf64>, memref<1000xf64>, memref<1000xf64>) outs(%36 : memref<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %58 = arith.subf %arg7, %arg8 : f64
      %59 = math.exp %58 : f64
      %60 = arith.mulf %arg9, %59 : f64
      linalg.yield %60 : f64
    }
    %37 = memref.alloc() : memref<1000xf64>
    linalg.copy(%6, %37) : memref<1000xf64>, memref<1000xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%15, %16, %35 : memref<1000x200xf64>, memref<1000xf64>, memref<1000xf64>) outs(%37 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %58 = arith.subf %arg7, %arg8 : f64
      %59 = math.exp %58 : f64
      %60 = arith.mulf %arg9, %59 : f64
      %61 = arith.negf %60 : f64
      %62 = arith.addf %61, %arg10 : f64
      linalg.yield %62 : f64
    }
    %38 = memref.alloc() : memref<1000xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%34, %37 : memref<1000xf64>, memref<1000xf64>) outs(%38 : memref<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %39 = memref.alloc() : memref<1000x200xf64>
    linalg.copy(%3, %39) : memref<1000x200xf64>, memref<1000x200xf64> 
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%15, %38 : memref<1000x200xf64>, memref<1000xf64>) outs(%39 : memref<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.cmpf ogt, %arg7, %arg9 : f64
      %59 = select %58, %arg8, %cst : f64
      linalg.yield %59 : f64
    }
    %40 = memref.alloc() : memref<1000x200xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%36, %39 : memref<1000x200xf64>, memref<1000x200xf64>) outs(%40 : memref<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %41 = memref.alloc() : memref<200xf64>
    linalg.copy(%4, %41) : memref<200xf64>, memref<200xf64> 
    linalg.generic {indexing_maps = [#map10, #map0, #map0, #map10], iterator_types = ["parallel", "parallel"]} ins(%14, %13, %40 : memref<200xf64>, memref<1000x200xf64>, memref<1000x200xf64>) outs(%41 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %58 = arith.addf %arg9, %arg10 : f64
      linalg.yield %58 : f64
    }
    %42 = memref.alloc() : memref<1000x200xf64>
    linalg.generic {indexing_maps = [#map10, #map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%14, %13, %40 : memref<200xf64>, memref<1000x200xf64>, memref<1000x200xf64>) outs(%42 : memref<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %58 = arith.negf %arg9 : f64
      %59 = arith.mulf %58, %cst_1 : f64
      linalg.yield %59 : f64
    }
    %43 = memref.alloc() : memref<200xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%28, %41 : memref<200xf64>, memref<200xf64>) outs(%43 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %44 = memref.alloc() : memref<200xf64>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%30, %41 : memref<200xf64>, memref<200xf64>) outs(%44 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %45 = memref.alloc() : memref<1000x200x128xf64>
    linalg.generic {indexing_maps = [#map4, #map8, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %42 : memref<1000x200x128xf64>, memref<1000x200xf64>) outs(%45 : memref<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      linalg.yield %arg8 : f64
    }
    %46 = memref.alloc() : memref<200x128xf64>
    linalg.copy(%0, %46) : memref<200x128xf64>, memref<200x128xf64> 
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %10, %11, %45 : memref<200x128xf64>, memref<1000x200x128xf64>, memref<1000x200x128xf64>, memref<1000x200x128xf64>) outs(%46 : memref<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
      %58 = arith.mulf %arg7, %arg8 : f64
      %59 = arith.addf %58, %arg9 : f64
      %60 = arith.mulf %arg10, %59 : f64
      %61 = arith.mulf %arg10, %59 : f64
      %62 = arith.addf %60, %61 : f64
      %63 = arith.mulf %62, %arg8 : f64
      %64 = arith.addf %63, %arg11 : f64
      linalg.yield %64 : f64
    }
    %47 = memref.alloc() : memref<200x128xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%33, %46 : memref<200x128xf64>, memref<200x128xf64>) outs(%47 : memref<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %48 = memref.alloc() : memref<1000x200x128xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %10, %11, %45 : memref<200x128xf64>, memref<1000x200x128xf64>, memref<1000x200x128xf64>, memref<1000x200x128xf64>) outs(%48 : memref<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
      %58 = arith.mulf %arg7, %arg8 : f64
      %59 = arith.addf %58, %arg9 : f64
      %60 = arith.mulf %arg10, %59 : f64
      %61 = arith.mulf %arg10, %59 : f64
      %62 = arith.addf %60, %61 : f64
      %63 = arith.mulf %62, %arg7 : f64
      linalg.yield %63 : f64
    }
    %49 = memref.alloc() : memref<1000x200x128xf64>
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %10, %11, %45 : memref<200x128xf64>, memref<1000x200x128xf64>, memref<1000x200x128xf64>, memref<1000x200x128xf64>) outs(%49 : memref<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
      %58 = arith.mulf %arg7, %arg8 : f64
      %59 = arith.addf %58, %arg9 : f64
      %60 = arith.mulf %arg10, %59 : f64
      %61 = arith.mulf %arg10, %59 : f64
      %62 = arith.addf %60, %61 : f64
      linalg.yield %62 : f64
    }
    %50 = memref.alloc() : memref<200x128x128xf64>
    linalg.copy(%2, %50) : memref<200x128x128xf64>, memref<200x128x128xf64> 
    linalg.generic {indexing_maps = [#map5, #map6, #map7, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3, %10, %49 : memref<200x128x128xf64>, memref<1000x200x128xf64>, memref<1000x200x128xf64>) outs(%50 : memref<200x128x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %58 = arith.mulf %arg9, %arg8 : f64
      %59 = arith.addf %58, %arg10 : f64
      linalg.yield %59 : f64
    }
    %51 = memref.alloc() : memref<200x128x128xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %50 : memref<200x128x128xf64>, memref<200x128x128xf64>) outs(%51 : memref<200x128x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %52 = memref.alloc() : memref<1000x200x128xf64>
    linalg.copy(%1, %52) : memref<1000x200x128xf64>, memref<1000x200x128xf64> 
    linalg.generic {indexing_maps = [#map5, #map6, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3, %10, %49 : memref<200x128x128xf64>, memref<1000x200x128xf64>, memref<1000x200x128xf64>) outs(%52 : memref<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %58 = arith.mulf %arg9, %arg7 : f64
      %59 = arith.addf %58, %arg10 : f64
      linalg.yield %59 : f64
    }
    %53 = memref.alloc() : memref<1000x200x128xf64>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %52 : memref<1000x200x128xf64>, memref<1000x200x128xf64>) outs(%53 : memref<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %54 = memref.alloc() : memref<200x128xf64>
    linalg.copy(%0, %54) : memref<200x128xf64>, memref<200x128xf64> 
    linalg.generic {indexing_maps = [#map2, #map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1, %53 : memref<1000x128xf64>, memref<200x128xf64>, memref<1000x200x128xf64>) outs(%54 : memref<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %58 = arith.negf %arg9 : f64
      %59 = arith.addf %58, %arg10 : f64
      linalg.yield %59 : f64
    }
    %55 = memref.alloc() : memref<200x128xf64>
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %44 : memref<200x128xf64>, memref<200xf64>) outs(%55 : memref<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      linalg.yield %arg8 : f64
    }
    %56 = memref.alloc() : memref<200x128xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%47, %8 : memref<200x128xf64>, memref<200x128xf64>) outs(%56 : memref<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    %57 = memref.alloc() : memref<200x128xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%55, %56 : memref<200x128xf64>, memref<200x128xf64>) outs(%57 : memref<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %58 = arith.addf %arg7, %arg8 : f64
      linalg.yield %58 : f64
    }
    return %43, %54, %57, %51 : memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x128x128xf64>
  }
  func @lagrad_gmm_full(%arg0: memref<200xf64>, %arg1: memref<200x128xf64>, %arg2: memref<200x128xf64>, %arg3: memref<200x128x128xf64>, %arg4: memref<1000x128xf64>, %arg5: f64, %arg6: i64) -> (memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x128x128xf64>) {
    %0:4 = call @__grad_lagrad_gmm_objective_full(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x128x128xf64>, memref<1000x128xf64>, f64, i64) -> (memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x128x128xf64>)
    return %0#0, %0#1, %0#2, %0#3 : memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x128x128xf64>
  }
}

