// Made from running gmm_adbench_lagrad.mlir through the bufferization script,
// including buffer hoisting, stack promotion, and deallocation.
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

memref.global "private" constant @__constant_xf64_0 : memref<f64> = dense<1.000000e+03>
memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
memref.global "private" constant @__constant_1000xf64 : memref<{{n}}xf64> = dense<0.000000e+00>
memref.global "private" constant @__constant_1000x200xf64 : memref<{{n}}x{{k}}xf64> = dense<0.000000e+00>
memref.global "private" constant @__constant_1000x200x128xf64 : memref<{{n}}x{{k}}x{{d}}xf64> = dense<0.000000e+00>
memref.global "private" constant @__constant_200xf64 : memref<{{k}}xf64> = dense<0.000000e+00>
func @enzyme_gmm_objective_full(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>, %arg2: memref<{{k}}x{{d}}xf64>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64>, %arg4: memref<{{n}}x{{d}}xf64>, %arg5: f64, %arg6: i64, %out: memref<f64>) -> f64 {
  %cst = arith.constant 5.000000e-01 : f64
  %0 = memref.alloc() : memref<{{k}}x{{d}}xf64>
  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%0 : memref<{{k}}x{{d}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %36 = math.exp %arg7 : f64
    linalg.yield %36 : f64
  }
  %1 = memref.get_global @__constant_200xf64 : memref<{{k}}xf64>
  %2 = memref.alloc() : memref<{{k}}xf64>
  linalg.copy(%1, %2) : memref<{{k}}xf64>, memref<{{k}}xf64> 
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%2 : memref<{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %36 = arith.addf %arg7, %arg8 : f64
    linalg.yield %36 : f64
  }
  %3 = memref.alloc() : memref<{{n}}x{{k}}x{{d}}xf64>
  linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1 : memref<{{n}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) outs(%3 : memref<{{n}}x{{k}}x{{d}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %36 = arith.subf %arg7, %arg8 : f64
    linalg.yield %36 : f64
  }
  %4 = memref.get_global @__constant_1000x200x128xf64 : memref<{{n}}x{{k}}x{{d}}xf64>
  %5 = memref.alloc() : memref<{{n}}x{{k}}x{{d}}xf64>
  linalg.copy(%4, %5) : memref<{{n}}x{{k}}x{{d}}xf64>, memref<{{n}}x{{k}}x{{d}}xf64> 
  linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg3, %3 : memref<{{k}}x{{d}}x{{d}}xf64>, memref<{{n}}x{{k}}x{{d}}xf64>) outs(%5 : memref<{{n}}x{{k}}x{{d}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %36 = arith.mulf %arg7, %arg8 : f64
    %37 = arith.addf %36, %arg9 : f64
    linalg.yield %37 : f64
  }
  %6 = memref.alloc() : memref<{{n}}x{{k}}x{{d}}xf64>
  linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %3, %5 : memref<{{k}}x{{d}}xf64>, memref<{{n}}x{{k}}x{{d}}xf64>, memref<{{n}}x{{k}}x{{d}}xf64>) outs(%6 : memref<{{n}}x{{k}}x{{d}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
    %36 = arith.mulf %arg7, %arg8 : f64
    %37 = arith.addf %36, %arg9 : f64
    %38 = arith.mulf %37, %37 : f64
    linalg.yield %38 : f64
  }
  memref.dealloc %5 : memref<{{n}}x{{k}}x{{d}}xf64>
  memref.dealloc %3 : memref<{{n}}x{{k}}x{{d}}xf64>
  %7 = memref.get_global @__constant_1000x200xf64 : memref<{{n}}x{{k}}xf64>
  %8 = memref.alloc() : memref<{{n}}x{{k}}xf64>
  linalg.copy(%7, %8) : memref<{{n}}x{{k}}xf64>, memref<{{n}}x{{k}}xf64> 
  linalg.generic {indexing_maps = [#map4, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6 : memref<{{n}}x{{k}}x{{d}}xf64>) outs(%8 : memref<{{n}}x{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %36 = arith.addf %arg7, %arg8 : f64
    linalg.yield %36 : f64
  }
  memref.dealloc %6 : memref<{{n}}x{{k}}x{{d}}xf64>
  %9 = memref.alloc() : memref<{{k}}xf64>
  linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%arg0, %2 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%9 : memref<{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %36 = arith.addf %arg7, %arg8 : f64
    linalg.yield %36 : f64
  }
  %10 = memref.alloc() : memref<{{n}}x{{k}}xf64>
  linalg.generic {indexing_maps = [#map10, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%9, %8 : memref<{{k}}xf64>, memref<{{n}}x{{k}}xf64>) outs(%10 : memref<{{n}}x{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %36 = arith.mulf %arg8, %cst : f64
    %37 = arith.subf %arg7, %36 : f64
    linalg.yield %37 : f64
  }
  memref.dealloc %9 : memref<{{k}}xf64>
  memref.dealloc %8 : memref<{{n}}x{{k}}xf64>
  %11 = memref.get_global @__constant_1000xf64 : memref<{{n}}xf64>
  %12 = memref.alloc() : memref<{{n}}xf64>
  linalg.copy(%11, %12) : memref<{{n}}xf64>, memref<{{n}}xf64> 
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%10 : memref<{{n}}x{{k}}xf64>) outs(%12 : memref<{{n}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %36 = arith.cmpf ogt, %arg7, %arg8 : f64
    %37 = scf.if %36 -> (f64) {
      scf.yield %arg7 : f64
    } else {
      scf.yield %arg8 : f64
    }
    linalg.yield %37 : f64
  }
  %13 = memref.get_global @__constant_1000xf64 : memref<{{n}}xf64>
  %14 = memref.alloc() : memref<{{n}}xf64>
  linalg.copy(%13, %14) : memref<{{n}}xf64>, memref<{{n}}xf64> 
  linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%10, %12 : memref<{{n}}x{{k}}xf64>, memref<{{n}}xf64>) outs(%14 : memref<{{n}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %36 = arith.subf %arg7, %arg8 : f64
    %37 = math.exp %36 : f64
    %38 = arith.addf %37, %arg9 : f64
    linalg.yield %38 : f64
  }
  memref.dealloc %10 : memref<{{n}}x{{k}}xf64>
  %15 = memref.alloc() : memref<{{n}}xf64>
  linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel"]} ins(%14 : memref<{{n}}xf64>) outs(%15 : memref<{{n}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %36 = math.log %arg7 : f64
    linalg.yield %36 : f64
  }
  memref.dealloc %14 : memref<{{n}}xf64>
  %16 = memref.alloc() : memref<{{n}}xf64>
  linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%15, %12 : memref<{{n}}xf64>, memref<{{n}}xf64>) outs(%16 : memref<{{n}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %36 = arith.addf %arg7, %arg8 : f64
    linalg.yield %36 : f64
  }
  memref.dealloc %15 : memref<{{n}}xf64>
  memref.dealloc %12 : memref<{{n}}xf64>
  %17 = memref.get_global @__constant_xf64 : memref<f64>
  %18 = memref.alloca() : memref<f64>
  linalg.copy(%17, %18) : memref<f64>, memref<f64> 
  linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%16 : memref<{{n}}xf64>) outs(%18 : memref<f64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %36 = arith.addf %arg7, %arg8 : f64
    linalg.yield %36 : f64
  }
  memref.dealloc %16 : memref<{{n}}xf64>
  %19 = memref.get_global @__constant_200xf64 : memref<{{k}}xf64>
  %20 = memref.alloc() : memref<{{k}}xf64>
  linalg.copy(%19, %20) : memref<{{k}}xf64>, memref<{{k}}xf64> 
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%0 : memref<{{k}}x{{d}}xf64>) outs(%20 : memref<{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %36 = arith.mulf %arg7, %arg7 : f64
    %37 = arith.addf %36, %arg8 : f64
    linalg.yield %37 : f64
  }
  memref.dealloc %0 : memref<{{k}}x{{d}}xf64>
  %21 = memref.get_global @__constant_200xf64 : memref<{{k}}xf64>
  %22 = memref.alloc() : memref<{{k}}xf64>
  linalg.copy(%21, %22) : memref<{{k}}xf64>, memref<{{k}}xf64> 
  linalg.generic {indexing_maps = [#map4, #map12], iterator_types = ["parallel", "reduction", "reduction"]} ins(%arg3 : memref<{{k}}x{{d}}x{{d}}xf64>) outs(%22 : memref<{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %36 = arith.mulf %arg7, %arg7 : f64
    %37 = arith.addf %36, %arg8 : f64
    linalg.yield %37 : f64
  }
  %23 = memref.alloc() : memref<{{k}}xf64>
  linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%20, %22 : memref<{{k}}xf64>, memref<{{k}}xf64>) outs(%23 : memref<{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %36 = arith.addf %arg7, %arg8 : f64
    linalg.yield %36 : f64
  }
  memref.dealloc %22 : memref<{{k}}xf64>
  memref.dealloc %20 : memref<{{k}}xf64>
  %24 = memref.get_global @__constant_xf64 : memref<f64>
  %25 = memref.alloca() : memref<f64>
  linalg.copy(%24, %25) : memref<f64>, memref<f64> 
  linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%23 : memref<{{k}}xf64>) outs(%25 : memref<f64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %36 = arith.mulf %cst, %arg5 : f64
    %37 = arith.mulf %36, %arg5 : f64
    %38 = arith.mulf %37, %arg7 : f64
    %39 = arith.addf %38, %arg8 : f64
    linalg.yield %39 : f64
  }
  memref.dealloc %23 : memref<{{k}}xf64>
  %26 = memref.get_global @__constant_xf64 : memref<f64>
  %27 = memref.alloca() : memref<f64>
  linalg.copy(%26, %27) : memref<f64>, memref<f64> 
  linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%2 : memref<{{k}}xf64>) outs(%27 : memref<f64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %36 = arith.sitofp %arg6 : i64 to f64
    %37 = arith.mulf %36, %arg7 : f64
    %38 = arith.addf %37, %arg8 : f64
    linalg.yield %38 : f64
  }
  memref.dealloc %2 : memref<{{k}}xf64>
  %28 = memref.alloca() : memref<f64>
  linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%25, %27 : memref<f64>, memref<f64>) outs(%28 : memref<f64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %36 = arith.subf %arg7, %arg8 : f64
    linalg.yield %36 : f64
  }
  %29 = memref.get_global @__constant_xf64 : memref<f64>
  %30 = memref.alloca() : memref<f64>
  linalg.copy(%29, %30) : memref<f64>, memref<f64> 
  linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%arg0 : memref<{{k}}xf64>) outs(%30 : memref<f64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %36 = math.exp %arg7 : f64
    %37 = arith.addf %36, %arg8 : f64
    linalg.yield %37 : f64
  }
  %31 = memref.alloca() : memref<f64>
  linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%30 : memref<f64>) outs(%31 : memref<f64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %36 = math.log %arg7 : f64
    linalg.yield %36 : f64
  }
  %32 = memref.get_global @__constant_xf64_0 : memref<f64>
  %33 = memref.alloca() : memref<f64>
  linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%32, %31 : memref<f64>, memref<f64>) outs(%33 : memref<f64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %36 = arith.mulf %arg7, %arg8 : f64
    linalg.yield %36 : f64
  }
  %34 = memref.alloca() : memref<f64>
  linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%18, %33 : memref<f64>, memref<f64>) outs(%34 : memref<f64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %36 = arith.subf %arg7, %arg8 : f64
    linalg.yield %36 : f64
  }
  %35 = memref.alloca() : memref<f64>
  linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%34, %28 : memref<f64>, memref<f64>) outs(%35 : memref<f64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %36 = arith.addf %arg7, %arg8 : f64
    linalg.yield %36 : f64
  }
  linalg.copy(%35, %out) : memref<f64>, memref<f64>

  %ret = arith.constant 0.0 : f64
  return %ret : f64
}

func @enzyme_gmm_full(
  %alphas: memref<{{k}}xf64>,
  %means: memref<{{k}}x{{d}}xf64>,
  %Qs: memref<{{k}}x{{d}}xf64>,
  %Ls: memref<{{k}}x{{d}}x{{d}}xf64>,
  %x: memref<{{n}}x{{d}}xf64>,
  %wishart_gamma: f64,
  %wishart_m: i64
) -> (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>) {
  %dalphas = memref.alloc() : memref<{{k}}xf64>
  %dmeans = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %dQs = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %dLs = memref.alloc() : memref<{{k}}x{{d}}x{{d}}xf64>
  %zero = arith.constant 0.0 : f64
  linalg.fill(%zero, %dalphas) : f64, memref<{{k}}xf64>
  linalg.fill(%zero, %dmeans) : f64, memref<{{k}}x{{d}}xf64>
  linalg.fill(%zero, %dQs) : f64, memref<{{k}}x{{d}}xf64>
  linalg.fill(%zero, %dLs) : f64, memref<{{k}}x{{d}}x{{d}}xf64>

  %out = memref.alloca() : memref<f64>
  %dout = memref.alloca() : memref<f64>
  memref.store %zero, %out[] : memref<f64>
  %one = arith.constant 1.0 : f64
  memref.store %one, %dout[] : memref<f64>

  %f = constant @enzyme_gmm_objective_full : (
    memref<{{k}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}x{{d}}xf64>,
    memref<{{n}}x{{d}}xf64>,
    f64,
    i64,
    memref<f64>
  ) -> f64
  %df = standalone.diff %f {const = [4]} : (
    memref<{{k}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}x{{d}}xf64>,
    memref<{{n}}x{{d}}xf64>,
    f64,
    i64,
    memref<f64>
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
    i64,
    memref<f64>,
    memref<f64>
  ) -> f64
  call_indirect %df(%alphas, %dalphas, %means, %dmeans, %Qs, %dQs, %Ls, %dLs, %x, %wishart_gamma, %wishart_m, %out, %dout) : (
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
    i64,
    memref<f64>,
    memref<f64>
  ) -> f64
  return %dalphas, %dmeans, %dQs, %dLs : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>
}
