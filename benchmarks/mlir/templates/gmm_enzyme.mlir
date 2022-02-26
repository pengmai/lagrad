// This was taken by bufferizing gmm.mlir. It's based on the memory-optimized Enzyme implementation,
// but done to include the overhead of bufferization when put through enzyme.

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0)[s0] -> (d0 + s0)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0) -> ()>
module  {
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_10xf64 : memref<{{d}}xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_25xf64 : memref<{{k}}xf64> = dense<0.000000e+00>
  func @enzyme_gmm_opt_compressed(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x10xf64>, %arg2: memref<{{k}}x10xf64>, %arg3: memref<{{k}}x45xf64>, %arg4: memref<1000x10xf64>, %arg5: f64, %arg6: i64) -> f64 {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = memref.alloc() : memref<{{k}}x10xf64>
    memref.dealloc %0 : memref<{{k}}x10xf64>
    %1 = memref.get_global @__constant_25xf64 : memref<{{k}}xf64>
    %2 = memref.get_global @__constant_10xf64 : memref<{{d}}xf64>
    %3 = memref.alloca() : memref<{{d}}xf64>
    %4 = memref.alloca() : memref<{{k}}xf64>
    %5 = memref.get_global @__constant_xf64 : memref<f64>
    %6 = memref.alloca() : memref<f64>
    %7 = memref.alloc() : memref<{{k}}x10xf64>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<{{k}}x10xf64>) outs(%7 : memref<{{k}}x10xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = math.exp %arg7 : f64
      linalg.yield %32 : f64
    }
    %8 = memref.alloca() : memref<{{k}}xf64>
    linalg.copy(%1, %8) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<{{k}}x10xf64>) outs(%8 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = arith.addf %arg7, %arg8 : f64
      linalg.yield %32 : f64
    }
    %cst_0 = arith.constant 5.000000e-01 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1000 = arith.constant {{n}} : index
    %c25 = arith.constant {{k}} : index
    %c10 = arith.constant {{d}} : index
    %9 = memref.alloca() : memref<{{d}}xf64>
    %10 = memref.alloca() : memref<{{d}}xf64>
    %11 = memref.alloca() : memref<{{d}}xf64>
    %12 = memref.alloca() : memref<f64>
    %13 = memref.alloca() : memref<f64>
    %14 = memref.alloca() : memref<f64>
    %15 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %cst) -> (f64) {
      %32 = scf.for %arg9 = %c0 to %c25 step %c1 iter_args(%arg10 = %4) -> (memref<{{k}}xf64>) {
        %39 = memref.subview %arg4[%arg7, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<{{d}}xf64, #map2>
        %40 = memref.cast %39 : memref<{{d}}xf64, #map2> to memref<{{d}}xf64>
        %41 = memref.subview %arg1[%arg9, 0] [1, 10] [1, 1] : memref<{{k}}x10xf64> to memref<{{d}}xf64, #map2>
        %42 = memref.cast %41 : memref<{{d}}xf64, #map2> to memref<{{d}}xf64>
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%40, %42 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%9 : memref<{{d}}xf64>) {
        ^bb0(%arg11: f64, %arg12: f64, %arg13: f64):  // no predecessors
          %54 = arith.subf %arg11, %arg12 : f64
          linalg.yield %54 : f64
        }
        %43 = memref.subview %7[%arg9, 0] [1, 10] [1, 1] : memref<{{k}}x10xf64> to memref<{{d}}xf64, #map2>
        %44 = memref.cast %43 : memref<{{d}}xf64, #map2> to memref<{{d}}xf64>
        %45 = memref.subview %arg3[%arg9, 0] [1, 45] [1, 1] : memref<{{k}}x45xf64> to memref<45xf64, #map2>
        %46 = memref.cast %45 : memref<45xf64, #map2> to memref<45xf64>
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%44, %9 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%10 : memref<{{d}}xf64>) {
        ^bb0(%arg11: f64, %arg12: f64, %arg13: f64):  // no predecessors
          %54 = arith.mulf %arg11, %arg12 : f64
          linalg.yield %54 : f64
        }
        linalg.fill(%cst, %3) : f64, memref<{{d}}xf64> 
        %47 = scf.for %arg11 = %c0 to %c10 step %c1 iter_args(%arg12 = %3) -> (memref<{{d}}xf64>) {
          %c20 = arith.constant 20 : index
          %54 = arith.subi %c20, %arg11 : index
          %55 = arith.subi %54, %c1 : index
          %56 = arith.muli %55, %arg11 : index
          %57 = arith.divsi %56, %c2 : index
          %58 = arith.addi %arg11, %c1 : index
          %59:2 = scf.for %arg13 = %58 to %c10 step %c1 iter_args(%arg14 = %arg12, %arg15 = %57) -> (memref<{{d}}xf64>, index) {
            %60 = memref.load %46[%arg15] : memref<45xf64>
            %61 = memref.load %9[%arg11] : memref<{{d}}xf64>
            %62 = memref.load %arg14[%arg13] : memref<{{d}}xf64>
            %63 = arith.mulf %60, %61 : f64
            %64 = arith.addf %63, %62 : f64
            memref.store %64, %arg14[%arg13] : memref<{{d}}xf64>
            %65 = arith.addi %arg15, %c1 : index
            scf.yield %arg14, %65 : memref<{{d}}xf64>, index
          }
          scf.yield %59#0 : memref<{{d}}xf64>
        }
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%10, %47 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%11 : memref<{{d}}xf64>) {
        ^bb0(%arg11: f64, %arg12: f64, %arg13: f64):  // no predecessors
          %54 = arith.addf %arg11, %arg12 : f64
          linalg.yield %54 : f64
        }
        linalg.copy(%5, %12) : memref<f64>, memref<f64> 
        linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["reduction"]} ins(%11 : memref<{{d}}xf64>) outs(%12 : memref<f64>) {
        ^bb0(%arg11: f64, %arg12: f64):  // no predecessors
          %54 = arith.mulf %arg11, %arg11 : f64
          %55 = arith.addf %54, %arg12 : f64
          linalg.yield %55 : f64
        }
        %48 = memref.load %12[] : memref<f64>
        %49 = arith.mulf %48, %cst_0 : f64
        %50 = memref.load %arg0[%arg9] : memref<{{k}}xf64>
        %51 = memref.load %8[%arg9] : memref<{{k}}xf64>
        %52 = arith.addf %50, %51 : f64
        %53 = arith.subf %52, %49 : f64
        memref.store %53, %arg10[%arg9] : memref<{{k}}xf64>
        scf.yield %arg10 : memref<{{k}}xf64>
      }
      %33 = memref.load %32[%c0] : memref<{{k}}xf64>
      memref.store %33, %6[] : memref<f64>
      linalg.copy(%6, %13) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["reduction"]} ins(%32 : memref<{{k}}xf64>) outs(%13 : memref<f64>) {
      ^bb0(%arg9: f64, %arg10: f64):  // no predecessors
        %39 = arith.cmpf ogt, %arg9, %arg10 : f64
        %40 = scf.if %39 -> (f64) {
          scf.yield %arg9 : f64
        } else {
          scf.yield %arg10 : f64
        }
        linalg.yield %40 : f64
      }
      %34 = memref.load %13[] : memref<f64>
      linalg.copy(%5, %14) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["reduction"]} ins(%32 : memref<{{k}}xf64>) outs(%14 : memref<f64>) {
      ^bb0(%arg9: f64, %arg10: f64):  // no predecessors
        %39 = arith.subf %arg9, %34 : f64
        %40 = math.exp %39 : f64
        %41 = arith.addf %40, %arg10 : f64
        linalg.yield %41 : f64
      }
      %35 = memref.load %14[] : memref<f64>
      %36 = math.log %35 : f64
      %37 = arith.addf %36, %34 : f64
      %38 = arith.addf %arg8, %37 : f64
      scf.yield %38 : f64
    }
    %16 = memref.load %arg0[%c0] : memref<{{k}}xf64>
    memref.store %16, %6[] : memref<f64>
    %17 = memref.alloca() : memref<f64>
    linalg.copy(%6, %17) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["reduction"]} ins(%arg0 : memref<{{k}}xf64>) outs(%17 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = arith.cmpf ogt, %arg7, %arg8 : f64
      %33 = scf.if %32 -> (f64) {
        scf.yield %arg7 : f64
      } else {
        scf.yield %arg8 : f64
      }
      linalg.yield %33 : f64
    }
    %18 = memref.load %17[] : memref<f64>
    %19 = memref.get_global @__constant_xf64 : memref<f64>
    %20 = memref.alloca() : memref<f64>
    linalg.copy(%19, %20) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["reduction"]} ins(%arg0 : memref<{{k}}xf64>) outs(%20 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %32 = arith.subf %arg7, %18 : f64
      %33 = math.exp %32 : f64
      %34 = arith.addf %33, %arg8 : f64
      linalg.yield %34 : f64
    }
    %21 = memref.load %20[] : memref<f64>
    %22 = math.log %21 : f64
    %23 = arith.addf %22, %18 : f64
    %cst_1 = arith.constant 1.000000e+03 : f64
    %24 = arith.mulf %cst_1, %23 : f64
    %c1_i64 = arith.constant 1 : i64
    %c10_i64 = arith.constant 10 : i64
    %25 = arith.addi %arg6, %c1_i64 : i64
    %26 = arith.addi %25, %c10_i64 : i64
    %27 = memref.alloca() : memref<f64>
    %28 = memref.alloca() : memref<f64>
    %29 = scf.for %arg7 = %c0 to %c25 step %c1 iter_args(%arg8 = %cst) -> (f64) {
      %32 = memref.subview %7[%arg7, 0] [1, 10] [1, 1] : memref<{{k}}x10xf64> to memref<{{d}}xf64, #map2>
      %33 = memref.cast %32 : memref<{{d}}xf64, #map2> to memref<{{d}}xf64>
      linalg.copy(%5, %27) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["reduction"]} ins(%33 : memref<{{d}}xf64>) outs(%27 : memref<f64>) {
      ^bb0(%arg9: f64, %arg10: f64):  // no predecessors
        %47 = arith.mulf %arg9, %arg9 : f64
        %48 = arith.addf %47, %arg10 : f64
        linalg.yield %48 : f64
      }
      %34 = memref.load %27[] : memref<f64>
      %35 = memref.subview %arg3[%arg7, 0] [1, 45] [1, 1] : memref<{{k}}x45xf64> to memref<45xf64, #map2>
      %36 = memref.cast %35 : memref<45xf64, #map2> to memref<45xf64>
      linalg.copy(%5, %28) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["reduction"]} ins(%36 : memref<45xf64>) outs(%28 : memref<f64>) {
      ^bb0(%arg9: f64, %arg10: f64):  // no predecessors
        %47 = arith.mulf %arg9, %arg9 : f64
        %48 = arith.addf %47, %arg10 : f64
        linalg.yield %48 : f64
      }
      %37 = memref.load %28[] : memref<f64>
      %38 = arith.addf %34, %37 : f64
      %39 = arith.mulf %arg5, %arg5 : f64
      %40 = arith.mulf %39, %cst_0 : f64
      %41 = arith.mulf %40, %38 : f64
      %42 = arith.sitofp %arg6 : i64 to f64
      %43 = memref.load %8[%arg7] : memref<{{k}}xf64>
      %44 = arith.mulf %42, %43 : f64
      %45 = arith.subf %41, %44 : f64
      %46 = arith.addf %arg8, %45 : f64
      scf.yield %46 : f64
    }
    memref.dealloc %7 : memref<{{k}}x10xf64>
    %30 = arith.subf %15, %24 : f64
    %31 = arith.addf %30, %29 : f64
    return %31 : f64
  }

  func @enzyme_gmm_opt_diff_compressed(
    %arg0: memref<{{k}}xf64>,
    %arg1: memref<{{k}}x{{d}}xf64>,
    %arg2: memref<{{k}}x{{d}}xf64>,
    %arg3: memref<{{k}}x{{tri_size}}xf64>,
    %arg4: memref<{{n}}x{{d}}xf64>,
    %arg5: f64,
    %arg6: i64
  ) -> (
    memref<{{k}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{tri_size}}xf64>
  ) {
    %zero = arith.constant 0.0 : f64
    %darg0 = memref.alloc() : memref<{{k}}xf64>
    %darg1 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    %darg2 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    %darg3 = memref.alloc() : memref<{{k}}x{{tri_size}}xf64>

    linalg.fill(%zero, %darg0) : f64, memref<{{k}}xf64>
    linalg.fill(%zero, %darg1) : f64, memref<{{k}}x{{d}}xf64>
    linalg.fill(%zero, %darg2) : f64, memref<{{k}}x{{d}}xf64>
    linalg.fill(%zero, %darg3) : f64, memref<{{k}}x{{tri_size}}xf64>

    %f = constant @enzyme_gmm_opt_compressed : (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{tri_size}}xf64>, memref<{{n}}x{{d}}xf64>, f64, i64) -> f64
    %df = standalone.diff %f {const = [4]} : (
      memref<{{k}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{tri_size}}xf64>,
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
      memref<{{k}}x{{tri_size}}xf64>,
      memref<{{k}}x{{tri_size}}xf64>,
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
      memref<{{k}}x{{tri_size}}xf64>,
      memref<{{k}}x{{tri_size}}xf64>,
      memref<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> f64
    return %darg0, %darg1, %darg2, %darg3 : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{tri_size}}xf64>
  }
}

