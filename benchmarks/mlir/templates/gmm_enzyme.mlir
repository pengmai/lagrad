// This was taken by bufferizing gmm.mlir. It's based on the memory-optimized Enzyme implementation,
// but done to include the overhead of bufferization when put through enzyme.

// This has been updated to use the comprehensive bufferized result of gmm.mlir.
#map0 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map1 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#map2 = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0) -> (d0)>
#map6 = affine_map<(d0)[s0] -> (d0 + s0)>
#map7 = affine_map<(d0) -> ()>
#map8 = affine_map<(d0, d1) -> ()>

memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_dxf64 : memref<{{d}}xf64> = dense<0.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_kxf64 : memref<{{k}}xf64> = dense<0.000000e+00> {alignment = 128 : i64}
func @enzyme_gmm_full_primal(%arg0: memref<{{k}}xf64, #map0>, %arg1: memref<{{k}}x{{d}}xf64, #map1>, %arg2: memref<{{k}}x{{d}}xf64, #map1>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64, #map2>, %arg4: memref<{{n}}x{{d}}xf64, #map1>, %arg5: f64, %arg6: i64, %out: memref<f64>) -> f64 {
  %cst = arith.constant 0.000000e+00 : f64
  %cst_0 = arith.constant 5.000000e-01 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1000 = arith.constant {{n}} : index
  %c25 = arith.constant {{k}} : index
  %cst_1 = arith.constant 1.000000e+03 : f64
  %0 = memref.get_global @__constant_xf64 : memref<f64>
  %1 = memref.get_global @__constant_dxf64 : memref<{{d}}xf64>
  %2 = memref.get_global @__constant_kxf64 : memref<{{k}}xf64>
  %3 = memref.alloc() {alignment = 128 : i64} : memref<{{k}}x{{d}}xf64>
  %4 = memref.alloc() {alignment = 128 : i64} : memref<{{k}}xf64>
  %5 = memref.alloc() {alignment = 128 : i64} : memref<f64>
  %6 = memref.alloc() {alignment = 128 : i64} : memref<{{k}}xf64>
  %7 = memref.alloc() {alignment = 128 : i64} : memref<{{d}}xf64>
  %8 = memref.alloc() {alignment = 128 : i64} : memref<{{d}}xf64>
  %9 = memref.alloc() {alignment = 128 : i64} : memref<{{d}}xf64>
  %10 = memref.alloc() {alignment = 128 : i64} : memref<{{d}}xf64>
  %11 = memref.alloc() {alignment = 128 : i64} : memref<f64>
  %12 = memref.alloc() {alignment = 128 : i64} : memref<f64>
  %13 = memref.alloc() {alignment = 128 : i64} : memref<f64>
  %14 = memref.alloc() {alignment = 128 : i64} : memref<f64>
  %15 = memref.alloc() {alignment = 128 : i64} : memref<f64>
  %16 = memref.alloc() {alignment = 128 : i64} : memref<f64>
  linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<{{k}}x{{d}}xf64, #map1>) outs(%3 : memref<{{k}}x{{d}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):
    %30 = math.exp %arg7 : f64
    linalg.yield %30 : f64
  }
  memref.copy %2, %6 : memref<{{k}}xf64> to memref<{{k}}xf64>
  linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<{{k}}x{{d}}xf64, #map1>) outs(%6 : memref<{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):
    %30 = arith.addf %arg7, %arg8 : f64
    linalg.yield %30 : f64
  }
  %17 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %cst) -> (f64) {
    %30 = memref.subview %arg4[%arg7, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64, #map1> to memref<{{d}}xf64, #map0>
    scf.for %arg9 = %c0 to %c25 step %c1 {
      %37 = memref.subview %arg1[%arg9, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64, #map1> to memref<{{d}}xf64, #map0>
      linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%30, %37 : memref<{{d}}xf64, #map0>, memref<{{d}}xf64, #map0>) outs(%7 : memref<{{d}}xf64>) {
      ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):
        %46 = arith.subf %arg10, %arg11 : f64
        linalg.yield %46 : f64
      }
      %38 = memref.subview %3[%arg9, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map6>
      %39 = memref.subview %arg3[%arg9, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : memref<{{k}}x{{d}}x{{d}}xf64, #map2> to memref<{{d}}x{{d}}xf64, #map1>
      linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%38, %7 : memref<{{d}}xf64, #map6>, memref<{{d}}xf64>) outs(%8 : memref<{{d}}xf64>) {
      ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):
        %46 = arith.mulf %arg10, %arg11 : f64
        linalg.yield %46 : f64
      }
      memref.copy %1, %9 : memref<{{d}}xf64> to memref<{{d}}xf64>
      linalg.matvec ins(%39, %7 : memref<{{d}}x{{d}}xf64, #map1>, memref<{{d}}xf64>) outs(%9 : memref<{{d}}xf64>)
      linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%8, %9 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%10 : memref<{{d}}xf64>) {
      ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):
        %46 = arith.addf %arg10, %arg11 : f64
        linalg.yield %46 : f64
      }
      memref.copy %0, %11 : memref<f64> to memref<f64>
      linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%10 : memref<{{d}}xf64>) outs(%11 : memref<f64>) {
      ^bb0(%arg10: f64, %arg11: f64):
        %46 = arith.mulf %arg10, %arg10 : f64
        %47 = arith.addf %46, %arg11 : f64
        linalg.yield %47 : f64
      }
      %40 = memref.load %11[] : memref<f64>
      %41 = arith.mulf %40, %cst_0 : f64
      %42 = memref.load %arg0[%arg9] : memref<{{k}}xf64, #map0>
      %43 = memref.load %6[%arg9] : memref<{{k}}xf64>
      %44 = arith.addf %42, %43 : f64
      %45 = arith.subf %44, %41 : f64
      memref.store %45, %4[%arg9] : memref<{{k}}xf64>
    }
    %31 = memref.load %4[%c0] : memref<{{k}}xf64>
    memref.store %31, %12[] : memref<f64>
    linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%4 : memref<{{k}}xf64>) outs(%12 : memref<f64>) {
    ^bb0(%arg9: f64, %arg10: f64):
      %37 = arith.cmpf ogt, %arg9, %arg10 : f64
      %38 = arith.select %37, %arg9, %arg10 : f64
      linalg.yield %38 : f64
    }
    %32 = memref.load %12[] : memref<f64>
    memref.copy %0, %13 : memref<f64> to memref<f64>
    linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%4 : memref<{{k}}xf64>) outs(%13 : memref<f64>) {
    ^bb0(%arg9: f64, %arg10: f64):
      %37 = arith.subf %arg9, %32 : f64
      %38 = math.exp %37 : f64
      %39 = arith.addf %38, %arg10 : f64
      linalg.yield %39 : f64
    }
    %33 = memref.load %13[] : memref<f64>
    %34 = math.log %33 : f64
    %35 = arith.addf %34, %32 : f64
    %36 = arith.addf %arg8, %35 : f64
    scf.yield %36 : f64
  }
  %18 = memref.load %arg0[%c0] : memref<{{k}}xf64, #map0>
  memref.store %18, %5[] : memref<f64>
  linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%arg0 : memref<{{k}}xf64, #map0>) outs(%5 : memref<f64>) {
  ^bb0(%arg7: f64, %arg8: f64):
    %30 = arith.cmpf ogt, %arg7, %arg8 : f64
    %31 = arith.select %30, %arg7, %arg8 : f64
    linalg.yield %31 : f64
  }
  %19 = memref.load %5[] : memref<f64>
  memref.copy %0, %14 : memref<f64> to memref<f64>
  linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%arg0 : memref<{{k}}xf64, #map0>) outs(%14 : memref<f64>) {
  ^bb0(%arg7: f64, %arg8: f64):
    %30 = arith.subf %arg7, %19 : f64
    %31 = math.exp %30 : f64
    %32 = arith.addf %31, %arg8 : f64
    linalg.yield %32 : f64
  }
  %20 = memref.load %14[] : memref<f64>
  %21 = math.log %20 : f64
  %22 = arith.addf %21, %19 : f64
  %23 = arith.mulf %22, %cst_1 : f64
  %24 = arith.mulf %arg5, %arg5 : f64
  %25 = arith.mulf %24, %cst_0 : f64
  %26 = arith.sitofp %arg6 : i64 to f64
  %27 = scf.for %arg7 = %c0 to %c25 step %c1 iter_args(%arg8 = %cst) -> (f64) {
    %30 = memref.subview %3[%arg7, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map6>
    memref.copy %0, %15 : memref<f64> to memref<f64>
    linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%30 : memref<{{d}}xf64, #map6>) outs(%15 : memref<f64>) {
    ^bb0(%arg9: f64, %arg10: f64):
      %40 = arith.mulf %arg9, %arg9 : f64
      %41 = arith.addf %40, %arg10 : f64
      linalg.yield %41 : f64
    }
    %31 = memref.load %15[] : memref<f64>
    %32 = memref.subview %arg3[%arg7, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : memref<{{k}}x{{d}}x{{d}}xf64, #map2> to memref<{{d}}x{{d}}xf64, #map1>
    memref.copy %0, %16 : memref<f64> to memref<f64>
    linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["reduction", "reduction"]} ins(%32 : memref<{{d}}x{{d}}xf64, #map1>) outs(%16 : memref<f64>) {
    ^bb0(%arg9: f64, %arg10: f64):
      %40 = arith.mulf %arg9, %arg9 : f64
      %41 = arith.addf %40, %arg10 : f64
      linalg.yield %41 : f64
    }
    %33 = memref.load %16[] : memref<f64>
    %34 = arith.addf %31, %33 : f64
    %35 = arith.mulf %25, %34 : f64
    %36 = memref.load %6[%arg7] : memref<{{k}}xf64>
    %37 = arith.mulf %26, %36 : f64
    %38 = arith.subf %35, %37 : f64
    %39 = arith.addf %arg8, %38 : f64
    scf.yield %39 : f64
  }
  %28 = arith.subf %17, %23 : f64
  %29 = arith.addf %28, %27 : f64
  memref.dealloc %16 : memref<f64>
  memref.dealloc %15 : memref<f64>
  memref.dealloc %14 : memref<f64>
  memref.dealloc %13 : memref<f64>
  memref.dealloc %12 : memref<f64>
  memref.dealloc %11 : memref<f64>
  memref.dealloc %10 : memref<{{d}}xf64>
  memref.dealloc %9 : memref<{{d}}xf64>
  memref.dealloc %8 : memref<{{d}}xf64>
  memref.dealloc %7 : memref<{{d}}xf64>
  memref.dealloc %6 : memref<{{k}}xf64>
  memref.dealloc %5 : memref<f64>
  memref.dealloc %4 : memref<{{k}}xf64>
  memref.dealloc %3 : memref<{{k}}x{{d}}xf64>

  memref.store %29, %out[] : memref<f64>
  return %cst : f64
}

func @enzyme_gmm_full(
  %alphas: memref<{{k}}xf64>,
  %dalphas: memref<{{k}}xf64>,
  %means: memref<{{k}}x{{d}}xf64>,
  %dmeans: memref<{{k}}x{{d}}xf64>,
  %Qs: memref<{{k}}x{{d}}xf64>,
  %dQs: memref<{{k}}x{{d}}xf64>,
  %Ls: memref<{{k}}x{{d}}x{{d}}xf64>,
  %dLs: memref<{{k}}x{{d}}x{{d}}xf64>,
  %x: memref<{{n}}x{{d}}xf64>,
  %wishart_gamma: f64,
  %wishart_m: i64
) {
  %zero = arith.constant 0.0 : f64
  %out = memref.alloca() : memref<f64>
  %dout = memref.alloca() : memref<f64>
  memref.store %zero, %out[] : memref<f64>
  %one = arith.constant 1.0 : f64
  memref.store %one, %dout[] : memref<f64>

  %f = constant @enzyme_gmm_full_primal : (
    memref<{{k}}xf64, #map0>,
    memref<{{k}}x{{d}}xf64, #map1>,
    memref<{{k}}x{{d}}xf64, #map1>,
    memref<{{k}}x{{d}}x{{d}}xf64, #map2>,
    memref<{{n}}x{{d}}xf64, #map1>,
    f64,
    i64,
    memref<f64>
  ) -> f64
  %df = standalone.diff %f {const = [4]} : (
    memref<{{k}}xf64, #map0>,
    memref<{{k}}x{{d}}xf64, #map1>,
    memref<{{k}}x{{d}}xf64, #map1>,
    memref<{{k}}x{{d}}x{{d}}xf64, #map2>,
    memref<{{n}}x{{d}}xf64, #map1>,
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
  return
}
