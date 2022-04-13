// This larger dataset GMM is segfaulting. Why? Is it making it to the end of the allocations?

#map0 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map1 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#map2 = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0) -> (d0)>
#map6 = affine_map<(d0)[s0] -> (d0 + s0)>
#map7 = affine_map<(d0) -> ()>
#map8 = affine_map<(d0, d1) -> ()>
#map9 = affine_map<(d0, d1) -> (d1)>
module {
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00> {alignment = 128 : i64}
  memref.global "private" constant @__constant_128xf64 : memref<128xf64> = dense<0.000000e+00> {alignment = 128 : i64}
  memref.global "private" constant @__constant_200xf64 : memref<200xf64> = dense<0.000000e+00> {alignment = 128 : i64}

  func private @print_memref_f64(memref<*xf64>) attributes { llvm.emit_c_interface }
  func @mlir_gmm_opt_full(%arg0: memref<200xf64, #map0>, %arg1: memref<200x128xf64, #map1>, %arg2: memref<200x128xf64, #map1>, %arg3: memref<200x128x128xf64, #map2>, %arg4: memref<1000x128xf64, #map1>, %arg5: f64, %arg6: i64) -> f64 {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e-01 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c200 = arith.constant 200 : index
    %cst_1 = arith.constant 1.000000e+03 : f64
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %1 = memref.get_global @__constant_128xf64 : memref<128xf64>
    %2 = memref.get_global @__constant_200xf64 : memref<200xf64>
    %3 = memref.alloc() {alignment = 128 : i64} : memref<200x128xf64>
    %4 = memref.alloc() {alignment = 128 : i64} : memref<200xf64>
    %5 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %6 = memref.alloc() {alignment = 128 : i64} : memref<200xf64>
    %7 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %8 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %9 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %10 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %11 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %12 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %13 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %14 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %15 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %16 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<200x128xf64, #map1>) outs(%3 : memref<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):
      %30 = math.exp %arg7 : f64
      linalg.yield %30 : f64
    }
    memref.copy %2, %6 : memref<200xf64> to memref<200xf64>
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<200x128xf64, #map1>) outs(%6 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):
      %30 = arith.addf %arg7, %arg8 : f64
      linalg.yield %30 : f64
    }
    %17 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %cst) -> (f64) {
      %30 = memref.subview %arg4[%arg7, 0] [1, 128] [1, 1] : memref<1000x128xf64, #map1> to memref<128xf64, #map0>
      scf.for %arg9 = %c0 to %c200 step %c1 {
        %37 = memref.subview %arg1[%arg9, 0] [1, 128] [1, 1] : memref<200x128xf64, #map1> to memref<128xf64, #map0>
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%30, %37 : memref<128xf64, #map0>, memref<128xf64, #map0>) outs(%7 : memref<128xf64>) {
        ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):
          %46 = arith.subf %arg10, %arg11 : f64
          linalg.yield %46 : f64
        }
        %38 = memref.subview %3[%arg9, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64, #map6>
        %39 = memref.subview %arg3[%arg9, 0, 0] [1, 128, 128] [1, 1, 1] : memref<200x128x128xf64, #map2> to memref<128x128xf64, #map1>
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%38, %7 : memref<128xf64, #map6>, memref<128xf64>) outs(%8 : memref<128xf64>) {
        ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):
          %46 = arith.mulf %arg10, %arg11 : f64
          linalg.yield %46 : f64
        }
        memref.copy %1, %9 : memref<128xf64> to memref<128xf64>
        linalg.matvec ins(%39, %7 : memref<128x128xf64, #map1>, memref<128xf64>) outs(%9 : memref<128xf64>)
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%8, %9 : memref<128xf64>, memref<128xf64>) outs(%10 : memref<128xf64>) {
        ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):
          %46 = arith.addf %arg10, %arg11 : f64
          linalg.yield %46 : f64
        }
        memref.copy %0, %11 : memref<f64> to memref<f64>
        linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%10 : memref<128xf64>) outs(%11 : memref<f64>) {
        ^bb0(%arg10: f64, %arg11: f64):
          %46 = arith.mulf %arg10, %arg10 : f64
          %47 = arith.addf %46, %arg11 : f64
          linalg.yield %47 : f64
        }
        %40 = memref.load %11[] : memref<f64>
        %41 = arith.mulf %40, %cst_0 : f64
        %42 = memref.load %arg0[%arg9] : memref<200xf64, #map0>
        %43 = memref.load %6[%arg9] : memref<200xf64>
        %44 = arith.addf %42, %43 : f64
        %45 = arith.subf %44, %41 : f64
        memref.store %45, %4[%arg9] : memref<200xf64>
      }
      %31 = memref.load %4[%c0] : memref<200xf64>
      memref.store %31, %12[] : memref<f64>
      linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%4 : memref<200xf64>) outs(%12 : memref<f64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %37 = arith.cmpf ogt, %arg9, %arg10 : f64
        %38 = arith.select %37, %arg9, %arg10 : f64
        linalg.yield %38 : f64
      }
      %32 = memref.load %12[] : memref<f64>
      memref.copy %0, %13 : memref<f64> to memref<f64>
      linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%4 : memref<200xf64>) outs(%13 : memref<f64>) {
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
    %18 = memref.load %arg0[%c0] : memref<200xf64, #map0>
    memref.store %18, %5[] : memref<f64>
    linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%arg0 : memref<200xf64, #map0>) outs(%5 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):
      %30 = arith.cmpf ogt, %arg7, %arg8 : f64
      %31 = arith.select %30, %arg7, %arg8 : f64
      linalg.yield %31 : f64
    }
    %19 = memref.load %5[] : memref<f64>
    memref.copy %0, %14 : memref<f64> to memref<f64>
    linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%arg0 : memref<200xf64, #map0>) outs(%14 : memref<f64>) {
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
    %27 = scf.for %arg7 = %c0 to %c200 step %c1 iter_args(%arg8 = %cst) -> (f64) {
      %30 = memref.subview %3[%arg7, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64, #map6>
      memref.copy %0, %15 : memref<f64> to memref<f64>
      linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%30 : memref<128xf64, #map6>) outs(%15 : memref<f64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %40 = arith.mulf %arg9, %arg9 : f64
        %41 = arith.addf %40, %arg10 : f64
        linalg.yield %41 : f64
      }
      %31 = memref.load %15[] : memref<f64>
      %32 = memref.subview %arg3[%arg7, 0, 0] [1, 128, 128] [1, 1, 1] : memref<200x128x128xf64, #map2> to memref<128x128xf64, #map1>
      memref.copy %0, %16 : memref<f64> to memref<f64>
      linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["reduction", "reduction"]} ins(%32 : memref<128x128xf64, #map1>) outs(%16 : memref<f64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %40 = arith.mulf %arg9, %arg9 : f64
        %41 = arith.addf %40, %arg10 : f64
        linalg.yield %41 : f64
      }
      %33 = memref.load %16[] : memref<f64>
      %34 = arith.addf %31, %33 : f64
      %35 = arith.mulf %25, %34 : f64
      %36 = memref.load %6[%arg7] : memref<200xf64>
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
    memref.dealloc %10 : memref<128xf64>
    memref.dealloc %9 : memref<128xf64>
    memref.dealloc %8 : memref<128xf64>
    memref.dealloc %7 : memref<128xf64>
    memref.dealloc %6 : memref<200xf64>
    memref.dealloc %5 : memref<f64>
    memref.dealloc %4 : memref<200xf64>
    memref.dealloc %3 : memref<200x128xf64>
    return %29 : f64
  }

  func @print_i(%arg0: index, %arg1: index) {
    %u = memref.alloca() : memref<2xf64>
    %0 = arith.index_cast %arg0 : index to i64
    %1 = arith.sitofp %0 : i64 to f64
    %2 = arith.index_cast %arg1 : index to i64
    %3 = arith.sitofp %2 : i64 to f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    memref.store %1, %u[%c0] : memref<2xf64>
    memref.store %3, %u[%c1] : memref<2xf64>

    %U = memref.cast %u : memref<2xf64> to memref<*xf64>
    call @print_memref_f64(%U) : (memref<*xf64>) -> ()
    return
  }

  func @__grad_mlir_gmm_opt_full(%arg0: memref<200xf64, #map0>, %arg1: memref<200x128xf64, #map1>, %arg2: memref<200x128xf64, #map1>, %arg3: memref<200x128x128xf64, #map2>, %arg4: memref<1000x128xf64, #map1>, %arg5: f64, %arg6: i64, %arg7: memref<200xf64, #map0>, %arg8: memref<200x128xf64, #map1>, %arg9: memref<200x128xf64, #map1>, %arg10: memref<200x128x128xf64, #map2>) {
    %cst = arith.constant 5.000000e-01 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c200 = arith.constant 200 : index
    %cst_0 = arith.constant 1.000000e+03 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %c199 = arith.constant 199 : index
    %cst_2 = arith.constant 0.000000e+00 : f64
    %c999 = arith.constant 999 : index

    %0 = memref.get_global @__constant_200xf64 : memref<200xf64>
    %1 = memref.get_global @__constant_xf64 : memref<f64>
    %2 = memref.get_global @__constant_128xf64 : memref<128xf64>
    %3 = memref.alloc() {alignment = 128 : i64} : memref<200x128xf64>
    %4 = memref.alloc() {alignment = 128 : i64} : memref<200xf64>
    %5 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %6 = memref.alloc() {alignment = 128 : i64} : memref<200xf64>
    %7 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %8 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %9 = memref.alloc() {alignment = 128 : i64} : memref<200x128xf64>
    %10 = memref.alloc() {alignment = 128 : i64} : memref<200xf64>
    %11 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %12 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %13 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %14 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf64>
    %15 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf64>
    %16 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %17 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %18 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %19 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %20 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %21 = memref.alloc() {alignment = 128 : i64} : memref<200xf64>
    %22 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %23 = memref.alloc() {alignment = 128 : i64} : memref<200xf64>
    %24 = memref.alloc() {alignment = 128 : i64} : memref<1000x128xf64>
    %25 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %26 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %27 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %28 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %29 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %30 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %31 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %32 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %33 = memref.alloc() {alignment = 128 : i64} : memref<200xf64>
    %34 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %35 = memref.alloc() {alignment = 128 : i64} : memref<200xf64>
    %36 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %37 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %38 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %39 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %40 = memref.alloc() {alignment = 128 : i64} : memref<f64>
    %41 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %42 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf64>
    %43 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %44 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %45 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %46 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf64>
    %47 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %48 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %49 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %50 = memref.alloc() {alignment = 128 : i64} : memref<128xf64>
    %51 = memref.alloc() {alignment = 128 : i64} : memref<200x128xf64>
    %52 = memref.alloc() {alignment = 128 : i64} : memref<200x128xf64>

    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<200x128xf64, #map1>) outs(%3 : memref<200x128xf64>) {
    ^bb0(%arg11: f64, %arg12: f64):
      %69 = math.exp %arg11 : f64
      linalg.yield %69 : f64
    }
    memref.copy %0, %6 : memref<200xf64> to memref<200xf64>
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<200x128xf64, #map1>) outs(%6 : memref<200xf64>) {
    ^bb0(%arg11: f64, %arg12: f64):
      %69 = arith.addf %arg11, %arg12 : f64
      linalg.yield %69 : f64
    }

    %53 = memref.load %arg0[%c0] : memref<200xf64, #map0>
    memref.store %53, %7[] : memref<f64>
    linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%arg0 : memref<200xf64, #map0>) outs(%7 : memref<f64>) {
    ^bb0(%arg11: f64, %arg12: f64):
      %69 = arith.cmpf ogt, %arg11, %arg12 : f64
      %70 = arith.select %69, %arg11, %arg12 : f64
      linalg.yield %70 : f64
    }
    %54 = memref.load %7[] : memref<f64>
    memref.copy %1, %8 : memref<f64> to memref<f64>
    linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%arg0 : memref<200xf64, #map0>) outs(%8 : memref<f64>) {
    ^bb0(%arg11: f64, %arg12: f64):
      %69 = arith.subf %arg11, %54 : f64
      %70 = math.exp %69 : f64
      %71 = arith.addf %70, %arg12 : f64
      linalg.yield %71 : f64
    }
    %55 = memref.load %8[] : memref<f64>
    %56 = arith.negf %cst_1 : f64
    linalg.fill(%cst_2, %9) : f64, memref<200x128xf64> 
    linalg.fill(%cst_2, %10) : f64, memref<200xf64> 
    %57 = arith.mulf %arg5, %arg5 : f64
    %58 = arith.mulf %57, %cst : f64
    %59 = arith.sitofp %arg6 : i64 to f64
    %60 = arith.mulf %56, %59 : f64

    %61 = scf.for %arg11 = %c0 to %c200 step %c1 iter_args(%arg12 = %cst_2) -> (f64) {
      %69 = arith.subi %c199, %arg11 : index
      %70 = memref.subview %3[%69, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64, #map6>
      memref.copy %1, %11 : memref<f64> to memref<f64>
      linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%70 : memref<128xf64, #map6>) outs(%11 : memref<f64>) {
      ^bb0(%arg13: f64, %arg14: f64):
        %83 = arith.mulf %arg13, %arg13 : f64
        %84 = arith.addf %83, %arg14 : f64
        linalg.yield %84 : f64
      }
      %71 = memref.load %11[] : memref<f64>
      %72 = memref.subview %arg3[%69, 0, 0] [1, 128, 128] [1, 1, 1] : memref<200x128x128xf64, #map2> to memref<128x128xf64, #map1>
      memref.copy %1, %12 : memref<f64> to memref<f64>
      linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["reduction", "reduction"]} ins(%72 : memref<128x128xf64, #map1>) outs(%12 : memref<f64>) {
      ^bb0(%arg13: f64, %arg14: f64):
        %83 = arith.mulf %arg13, %arg13 : f64
        %84 = arith.addf %83, %arg14 : f64
        linalg.yield %84 : f64
      }
      %73 = memref.load %12[] : memref<f64>
      %74 = arith.addf %71, %73 : f64
      %75 = memref.load %10[%69] : memref<200xf64>
      %76 = arith.addf %75, %60 : f64
      memref.store %76, %10[%69] : memref<200xf64>
      %77 = arith.mulf %74, %cst : f64
      %78 = arith.mulf %77, %arg5 : f64
      %79 = arith.addf %78, %arg12 : f64
      %80 = arith.addf %78, %79 : f64
      linalg.fill(%cst_2, %13) : f64, memref<f64> 
      memref.store %58, %13[] : memref<f64>
      linalg.generic {indexing_maps = [#map3, #map8, #map3], iterator_types = ["parallel", "parallel"]} ins(%72, %13 : memref<128x128xf64, #map1>, memref<f64>) outs(%14 : memref<128x128xf64>) {
      ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
        %83 = arith.mulf %arg14, %arg13 : f64
        %84 = arith.addf %83, %83 : f64
        linalg.yield %84 : f64
      }
      %81 = memref.subview %arg10[%69, 0, 0] [1, 128, 128] [1, 1, 1] : memref<200x128x128xf64, #map2> to memref<128x128xf64, #map1>
      linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%81, %14 : memref<128x128xf64, #map1>, memref<128x128xf64>) outs(%15 : memref<128x128xf64>) {
      ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
        %83 = arith.addf %arg13, %arg14 : f64
        linalg.yield %83 : f64
      }
      memref.copy %15, %81 : memref<128x128xf64> to memref<128x128xf64, #map1>
      linalg.fill(%cst_2, %16) : f64, memref<f64> 
      memref.store %58, %16[] : memref<f64>
      linalg.generic {indexing_maps = [#map5, #map7, #map5], iterator_types = ["parallel"]} ins(%70, %16 : memref<128xf64, #map6>, memref<f64>) outs(%17 : memref<128xf64>) {
      ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
        %83 = arith.mulf %arg14, %arg13 : f64
        %84 = arith.addf %83, %83 : f64
        linalg.yield %84 : f64
      }
      %82 = memref.subview %9[%69, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64, #map6>
      linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%82, %17 : memref<128xf64, #map6>, memref<128xf64>) outs(%18 : memref<128xf64>) {
      ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
        %83 = arith.addf %arg13, %arg14 : f64
        linalg.yield %83 : f64
      }
      memref.copy %18, %82 : memref<128xf64> to memref<128xf64, #map6>
      scf.yield %80 : f64
    }

    %62 = arith.mulf %56, %cst_0 : f64
    %63 = arith.divf %62, %55 : f64
    linalg.fill(%cst_2, %19) : f64, memref<f64> 
    memref.store %63, %19[] : memref<f64>
    memref.copy %1, %20 : memref<f64> to memref<f64>
    linalg.generic {indexing_maps = [#map5, #map7, #map7], iterator_types = ["parallel"]} ins(%arg0, %19 : memref<200xf64, #map0>, memref<f64>) outs(%20 : memref<f64>) {
    ^bb0(%arg11: f64, %arg12: f64, %arg13: f64):
      %69 = arith.subf %arg11, %54 : f64
      %70 = math.exp %69 : f64
      %71 = arith.mulf %arg12, %70 : f64
      %72 = arith.negf %71 : f64
      %73 = arith.addf %72, %arg13 : f64
      linalg.yield %73 : f64
    }
    %64 = memref.load %20[] : memref<f64>
    %65 = arith.addf %64, %62 : f64
    linalg.generic {indexing_maps = [#map5, #map7, #map5], iterator_types = ["parallel"]} ins(%arg0, %19 : memref<200xf64, #map0>, memref<f64>) outs(%21 : memref<200xf64>) {
    ^bb0(%arg11: f64, %arg12: f64, %arg13: f64):
      %69 = arith.subf %arg11, %54 : f64
      %70 = math.exp %69 : f64
      %71 = arith.mulf %arg12, %70 : f64
      linalg.yield %71 : f64
    }
    linalg.generic {doc = "Add in place", indexing_maps = [#map5, #map5], iterator_types = ["parallel"]} ins(%21 : memref<200xf64>) outs(%arg7 : memref<200xf64, #map0>) {
    ^bb0(%arg11: f64, %arg12: f64):
      %69 = arith.addf %arg11, %arg12 : f64
      linalg.yield %69 : f64
    }
    linalg.fill(%cst_2, %22) : f64, memref<f64> 
    memref.store %65, %22[] : memref<f64>
    memref.copy %0, %23 : memref<200xf64> to memref<200xf64>
    linalg.generic {indexing_maps = [#map5, #map7, #map5], iterator_types = ["parallel"]} ins(%arg0, %22 : memref<200xf64, #map0>, memref<f64>) outs(%23 : memref<200xf64>) {
    ^bb0(%arg11: f64, %arg12: f64, %arg13: f64):
      %69 = arith.cmpf ogt, %arg11, %arg13 : f64
      %70 = arith.select %69, %arg12, %cst_2 : f64
      linalg.yield %70 : f64
    }
    linalg.generic {doc = "Add in place", indexing_maps = [#map5, #map5], iterator_types = ["parallel"]} ins(%23 : memref<200xf64>) outs(%arg7 : memref<200xf64, #map0>) {
    ^bb0(%arg11: f64, %arg12: f64):
      %69 = arith.addf %arg11, %arg12 : f64
      linalg.yield %69 : f64
    }
    %66 = memref.load %22[] : memref<f64>
    %67 = memref.load %arg7[%c0] : memref<200xf64, #map0>
    %68 = arith.addf %67, %66 : f64
    memref.store %68, %arg7[%c0] : memref<200xf64, #map0>
    linalg.fill(%cst_2, %24) : f64, memref<1000x128xf64>
    scf.for %arg11 = %c0 to %c1000 step %c1 {
      %69 = arith.subi %c999, %arg11 : index
      %70 = memref.subview %arg4[%69, 0] [1, 128] [1, 1] : memref<1000x128xf64, #map1> to memref<128xf64, #map0>
      scf.for %arg12 = %c0 to %c200 step %c1 {
        %82 = memref.subview %arg1[%arg12, 0] [1, 128] [1, 1] : memref<200x128xf64, #map1> to memref<128xf64, #map0>
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%70, %82 : memref<128xf64, #map0>, memref<128xf64, #map0>) outs(%25 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %91 = arith.subf %arg13, %arg14 : f64
          linalg.yield %91 : f64
        }
        %83 = memref.subview %3[%arg12, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64, #map6>
        %84 = memref.subview %arg3[%arg12, 0, 0] [1, 128, 128] [1, 1, 1] : memref<200x128x128xf64, #map2> to memref<128x128xf64, #map1>
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%83, %25 : memref<128xf64, #map6>, memref<128xf64>) outs(%26 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %91 = arith.mulf %arg13, %arg14 : f64
          linalg.yield %91 : f64
        }
        memref.copy %2, %27 : memref<128xf64> to memref<128xf64>
        linalg.matvec ins(%84, %25 : memref<128x128xf64, #map1>, memref<128xf64>) outs(%27 : memref<128xf64>)
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%26, %27 : memref<128xf64>, memref<128xf64>) outs(%28 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %91 = arith.addf %arg13, %arg14 : f64
          linalg.yield %91 : f64
        }
        memref.copy %1, %29 : memref<f64> to memref<f64>
        linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%28 : memref<128xf64>) outs(%29 : memref<f64>) {
        ^bb0(%arg13: f64, %arg14: f64):
          %91 = arith.mulf %arg13, %arg13 : f64
          %92 = arith.addf %91, %arg14 : f64
          linalg.yield %92 : f64
        }
        %85 = memref.load %29[] : memref<f64>
        %86 = arith.mulf %85, %cst : f64
        %87 = memref.load %arg0[%arg12] : memref<200xf64, #map0>
        %88 = memref.load %6[%arg12] : memref<200xf64>
        %89 = arith.addf %87, %88 : f64
        %90 = arith.subf %89, %86 : f64
        memref.store %90, %4[%arg12] : memref<200xf64>
      }
      %71 = memref.load %4[%c0] : memref<200xf64>
      memref.store %71, %5[] : memref<f64>
      linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%4 : memref<200xf64>) outs(%5 : memref<f64>) {
      ^bb0(%arg12: f64, %arg13: f64):
        %82 = arith.cmpf ogt, %arg12, %arg13 : f64
        %83 = arith.select %82, %arg12, %arg13 : f64
        linalg.yield %83 : f64
      }
      %72 = memref.load %5[] : memref<f64>
      memref.copy %1, %30 : memref<f64> to memref<f64>
      linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction"]} ins(%4 : memref<200xf64>) outs(%30 : memref<f64>) {
      ^bb0(%arg12: f64, %arg13: f64):
        %82 = arith.subf %arg12, %72 : f64
        %83 = math.exp %82 : f64
        %84 = arith.addf %83, %arg13 : f64
        linalg.yield %84 : f64
      }
      %73 = memref.load %30[] : memref<f64>
      %74 = arith.divf %cst_1, %73 : f64
      linalg.fill(%cst_2, %31) : f64, memref<f64> 
      memref.store %74, %31[] : memref<f64>
      memref.copy %1, %32 : memref<f64> to memref<f64>
      linalg.generic {indexing_maps = [#map5, #map7, #map7], iterator_types = ["parallel"]} ins(%4, %31 : memref<200xf64>, memref<f64>) outs(%32 : memref<f64>) {
      ^bb0(%arg12: f64, %arg13: f64, %arg14: f64):
        %82 = arith.subf %arg12, %72 : f64
        %83 = math.exp %82 : f64
        %84 = arith.mulf %arg13, %83 : f64
        %85 = arith.negf %84 : f64
        %86 = arith.addf %85, %arg14 : f64
        linalg.yield %86 : f64
      }
      %75 = memref.load %32[] : memref<f64>
      %76 = arith.addf %75, %cst_1 : f64
      linalg.generic {indexing_maps = [#map5, #map7, #map5], iterator_types = ["parallel"]} ins(%4, %31 : memref<200xf64>, memref<f64>) outs(%33 : memref<200xf64>) {
      ^bb0(%arg12: f64, %arg13: f64, %arg14: f64):
        %82 = arith.subf %arg12, %72 : f64
        %83 = math.exp %82 : f64
        %84 = arith.mulf %arg13, %83 : f64
        linalg.yield %84 : f64
      }
      linalg.fill(%cst_2, %34) : f64, memref<f64> 
      memref.store %76, %34[] : memref<f64>
      memref.copy %0, %35 : memref<200xf64> to memref<200xf64>
      linalg.generic {indexing_maps = [#map5, #map7, #map5], iterator_types = ["parallel"]} ins(%4, %34 : memref<200xf64>, memref<f64>) outs(%35 : memref<200xf64>) {
      ^bb0(%arg12: f64, %arg13: f64, %arg14: f64):
        %82 = arith.cmpf ogt, %arg12, %arg14 : f64
        %83 = arith.select %82, %arg13, %cst_2 : f64
        linalg.yield %83 : f64
      }
      linalg.generic {doc = "Add in place", indexing_maps = [#map5, #map5], iterator_types = ["parallel"]} ins(%35 : memref<200xf64>) outs(%33 : memref<200xf64>) {
      ^bb0(%arg12: f64, %arg13: f64):
        %82 = arith.addf %arg12, %arg13 : f64
        linalg.yield %82 : f64
      }
      %77 = memref.load %34[] : memref<f64>
      %78 = memref.load %33[%c0] : memref<200xf64>
      %79 = arith.addf %78, %77 : f64
      memref.store %79, %33[%c0] : memref<200xf64>
      %80 = memref.subview %arg4[%69, 0] [1, 128] [1, 1] : memref<1000x128xf64, #map1> to memref<128xf64, #map0>
      %81 = memref.subview %24[%69, 0] [1, 128] [1, 1] : memref<1000x128xf64> to memref<128xf64, #map6>

      // This Upper bound is originally 200
      %ub = arith.constant 200 : index
      scf.for %arg12 = %c0 to %ub step %c1 {
        // call @print_i(%arg11, %arg12) : (index, index) -> ()
        %82 = arith.subi %c199, %arg12 : index
        %83 = memref.subview %arg1[%82, 0] [1, 128] [1, 1] : memref<200x128xf64, #map1> to memref<128xf64, #map0>
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%80, %83 : memref<128xf64, #map0>, memref<128xf64, #map0>) outs(%36 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %96 = arith.subf %arg13, %arg14 : f64
          linalg.yield %96 : f64
        }
        %84 = memref.subview %3[%82, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64, #map6>
        %85 = memref.subview %arg3[%82, 0, 0] [1, 128, 128] [1, 1, 1] : memref<200x128x128xf64, #map2> to memref<128x128xf64, #map1>
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%84, %36 : memref<128xf64, #map6>, memref<128xf64>) outs(%37 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %96 = arith.mulf %arg13, %arg14 : f64
          linalg.yield %96 : f64
        }
        memref.copy %2, %38 : memref<128xf64> to memref<128xf64>
        linalg.generic
          {doc = "copy", indexing_maps = [#map5, #map5], iterator_types = ["parallel"]}
          ins(%2 : memref<128xf64>)
          outs(%38 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64):
          linalg.yield %arg13 : f64
        }
        linalg.matvec ins(%85, %36 : memref<128x128xf64, #map1>, memref<128xf64>) outs(%38 : memref<128xf64>)
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%37, %38 : memref<128xf64>, memref<128xf64>) outs(%39 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %96 = arith.addf %arg13, %arg14 : f64
          linalg.yield %96 : f64
        }
        %86 = memref.load %33[%82] : memref<200xf64>
        %87 = arith.negf %86 : f64
        %88 = memref.load %10[%82] : memref<200xf64>
        %89 = arith.addf %88, %86 : f64
        memref.store %89, %10[%82] : memref<200xf64>
        %90 = memref.load %arg7[%82] : memref<200xf64, #map0>
        %91 = arith.addf %90, %86 : f64
        memref.store %91, %arg7[%82] : memref<200xf64, #map0>
        %92 = arith.mulf %87, %cst : f64
        linalg.fill(%cst_2, %40) : f64, memref<f64> 
        memref.store %92, %40[] : memref<f64>
        linalg.generic {indexing_maps = [#map5, #map7, #map5], iterator_types = ["parallel"]} ins(%39, %40 : memref<128xf64>, memref<f64>) outs(%41 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %96 = arith.mulf %arg14, %arg13 : f64
          %97 = arith.addf %96, %96 : f64
          linalg.yield %97 : f64
        }
        linalg.generic {doc = "Vector-vector outer product", indexing_maps = [#map4, #map9, #map3], iterator_types = ["parallel", "parallel"], library_call = "souter"} ins(%41, %36 : memref<128xf64>, memref<128xf64>) outs(%42 : memref<128x128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %96 = arith.mulf %arg13, %arg14 : f64
          linalg.yield %96 : f64
        }
        memref.copy %2, %43 : memref<128xf64> to memref<128xf64>
        linalg.generic {doc = "Vector-Matrix multiplication", indexing_maps = [#map4, #map3, #map9], iterator_types = ["reduction", "parallel"], library_call = "svecmat"} ins(%41, %85 : memref<128xf64>, memref<128x128xf64, #map1>) outs(%43 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %96 = arith.mulf %arg13, %arg14 : f64
          %97 = arith.addf %96, %arg15 : f64
          linalg.yield %97 : f64
        }
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%41, %36 : memref<128xf64>, memref<128xf64>) outs(%44 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %96 = arith.mulf %arg13, %arg14 : f64
          linalg.yield %96 : f64
        }
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%41, %84 : memref<128xf64>, memref<128xf64, #map6>) outs(%45 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %96 = arith.mulf %arg13, %arg14 : f64
          linalg.yield %96 : f64
        }
        linalg.generic {doc = "Add in place", indexing_maps = [#map5, #map5], iterator_types = ["parallel"]} ins(%45 : memref<128xf64>) outs(%43 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64):
          %96 = arith.addf %arg13, %arg14 : f64
          linalg.yield %96 : f64
        }
        %93 = memref.subview %arg10[%82, 0, 0] [1, 128, 128] [1, 1, 1] : memref<200x128x128xf64, #map2> to memref<128x128xf64, #map1>
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%93, %42 : memref<128x128xf64, #map1>, memref<128x128xf64>) outs(%46 : memref<128x128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %96 = arith.addf %arg13, %arg14 : f64
          linalg.yield %96 : f64
        }
        // call @print_i(%arg11, %arg12) : (index, index) -> ()
        // This appears to be the op that is segfaulting.
        // memref.copy %46, %93 : memref<128x128xf64> to memref<128x128xf64, #map1>
        linalg.generic {doc = "copy", indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]}
          ins(%46 : memref<128x128xf64>)
          outs(%93 : memref<128x128xf64, #map1>) {
        ^bb0(%arg13: f64, %arg14: f64):
          linalg.yield %arg13 : f64
        }
        // call @print_i(%arg11, %arg12) : (index, index) -> ()
        %94 = memref.subview %9[%82, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64, #map6>
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%94, %44 : memref<128xf64, #map6>, memref<128xf64>) outs(%47 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %96 = arith.addf %arg13, %arg14 : f64
          linalg.yield %96 : f64
        }
        // memref.copy %47, %94 : memref<128xf64> to memref<128xf64, #map6>
        linalg.generic
          {doc = "copy", indexing_maps = [#map5, #map5], iterator_types = ["parallel"]}
          ins(%47 : memref<128xf64>)
          outs(%94 : memref<128xf64, #map6>) {
        ^bb0(%arg13: f64, %arg14: f64):
          linalg.yield %arg13 : f64
        }

        linalg.generic {indexing_maps = [#map5, #map5, #map5, #map5], iterator_types = ["parallel"]} ins(%80, %83, %43 : memref<128xf64, #map0>, memref<128xf64, #map0>, memref<128xf64>) outs(%48 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64):
          %96 = arith.negf %arg15 : f64
          linalg.yield %96 : f64
        }
        %95 = memref.subview %arg8[%82, 0] [1, 128] [1, 1] : memref<200x128xf64, #map1> to memref<128xf64, #map0>
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%95, %48 : memref<128xf64, #map0>, memref<128xf64>) outs(%49 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %96 = arith.addf %arg13, %arg14 : f64
          linalg.yield %96 : f64
        }
        // memref.copy %49, %95 : memref<128xf64> to memref<128xf64, #map0>
        linalg.generic
          {doc = "copy", indexing_maps = [#map5, #map5], iterator_types = ["parallel"]}
          ins(%49 : memref<128xf64>)
          outs(%95 : memref<128xf64, #map0>) {
        ^bb0(%arg13: f64, %arg14: f64):
          linalg.yield %arg13 : f64
        }

        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel"]} ins(%81, %43 : memref<128xf64, #map6>, memref<128xf64>) outs(%50 : memref<128xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64):
          %96 = arith.addf %arg13, %arg14 : f64
          linalg.yield %96 : f64
        }
        // memref.copy %50, %81 : memref<128xf64> to memref<128xf64, #map6>
        linalg.generic
          {doc = "copy", indexing_maps = [#map5, #map5], iterator_types = ["parallel"]}
          ins(%50 : memref<128xf64>)
          outs(%81 : memref<128xf64, #map6>) {
        ^bb0(%arg13: f64, %arg14: f64):
          linalg.yield %arg13 : f64
        }
      }
    }

    linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg2, %10 : memref<200x128xf64, #map1>, memref<200xf64>) outs(%51 : memref<200x128xf64>) {
    ^bb0(%arg11: f64, %arg12: f64, %arg13: f64):
      linalg.yield %arg12 : f64
    }
    linalg.generic {doc = "Add in place", indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%51 : memref<200x128xf64>) outs(%arg9 : memref<200x128xf64, #map1>) {
    ^bb0(%arg11: f64, %arg12: f64):
      %69 = arith.addf %arg11, %arg12 : f64
      linalg.yield %69 : f64
    }
    linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg2, %9 : memref<200x128xf64, #map1>, memref<200x128xf64>) outs(%52 : memref<200x128xf64>) {
    ^bb0(%arg11: f64, %arg12: f64, %arg13: f64):
      %69 = math.exp %arg11 : f64
      %70 = arith.mulf %arg12, %69 : f64
      linalg.yield %70 : f64
    }
    linalg.generic {doc = "Add in place", indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%52 : memref<200x128xf64>) outs(%arg9 : memref<200x128xf64, #map1>) {
    ^bb0(%arg11: f64, %arg12: f64):
      %69 = arith.addf %arg11, %arg12 : f64
      linalg.yield %69 : f64
    }

    memref.dealloc %52 : memref<200x128xf64>
    memref.dealloc %51 : memref<200x128xf64>
    memref.dealloc %50 : memref<128xf64>
    memref.dealloc %49 : memref<128xf64>
    memref.dealloc %48 : memref<128xf64>
    memref.dealloc %47 : memref<128xf64>
    memref.dealloc %46 : memref<128x128xf64>
    memref.dealloc %45 : memref<128xf64>
    memref.dealloc %44 : memref<128xf64>
    memref.dealloc %43 : memref<128xf64>
    memref.dealloc %42 : memref<128x128xf64>
    memref.dealloc %41 : memref<128xf64>
    memref.dealloc %40 : memref<f64>
    memref.dealloc %39 : memref<128xf64>
    memref.dealloc %38 : memref<128xf64>
    memref.dealloc %37 : memref<128xf64>
    memref.dealloc %36 : memref<128xf64>
    memref.dealloc %35 : memref<200xf64>
    memref.dealloc %34 : memref<f64>
    memref.dealloc %33 : memref<200xf64>
    memref.dealloc %32 : memref<f64>
    memref.dealloc %31 : memref<f64>
    memref.dealloc %30 : memref<f64>
    memref.dealloc %29 : memref<f64>
    memref.dealloc %28 : memref<128xf64>
    memref.dealloc %27 : memref<128xf64>
    memref.dealloc %26 : memref<128xf64>
    memref.dealloc %25 : memref<128xf64>
    memref.dealloc %24 : memref<1000x128xf64>
    memref.dealloc %23 : memref<200xf64>
    memref.dealloc %22 : memref<f64>
    memref.dealloc %21 : memref<200xf64>
    memref.dealloc %20 : memref<f64>
    memref.dealloc %19 : memref<f64>
    memref.dealloc %18 : memref<128xf64>
    memref.dealloc %17 : memref<128xf64>
    memref.dealloc %16 : memref<f64>
    memref.dealloc %15 : memref<128x128xf64>
    memref.dealloc %14 : memref<128x128xf64>
    memref.dealloc %13 : memref<f64>
    memref.dealloc %12 : memref<f64>
    memref.dealloc %11 : memref<f64>
    memref.dealloc %10 : memref<200xf64>
    memref.dealloc %9 : memref<200x128xf64>
    memref.dealloc %8 : memref<f64>
    memref.dealloc %7 : memref<f64>
    memref.dealloc %6 : memref<200xf64>
    memref.dealloc %5 : memref<f64>
    memref.dealloc %4 : memref<200xf64>
    memref.dealloc %3 : memref<200x128xf64>
    return
  }
  func @lagrad_gmm_full(%arg0: memref<200xf64, #map0>, %arg1: memref<200xf64, #map0>, %arg2: memref<200x128xf64, #map1>, %arg3: memref<200x128xf64, #map1>, %arg4: memref<200x128xf64, #map1>, %arg5: memref<200x128xf64, #map1>, %arg6: memref<200x128x128xf64, #map2>, %arg7: memref<200x128x128xf64, #map2>, %arg8: memref<1000x128xf64, #map1>, %arg9: f64, %arg10: i64) {
    call @__grad_mlir_gmm_opt_full(%arg0, %arg2, %arg4, %arg6, %arg8, %arg9, %arg10, %arg1, %arg3, %arg5, %arg7) : (memref<200xf64, #map0>, memref<200x128xf64, #map1>, memref<200x128xf64, #map1>, memref<200x128x128xf64, #map2>, memref<1000x128xf64, #map1>, f64, i64, memref<200xf64, #map0>, memref<200x128xf64, #map1>, memref<200x128xf64, #map1>, memref<200x128x128xf64, #map2>) -> ()
    return
  }
}

