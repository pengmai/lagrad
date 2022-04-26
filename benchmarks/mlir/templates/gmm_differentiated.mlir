#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s0 + d1 * s2 + s1)>
#map5 = affine_map<(d0) -> ()>
module  {
  memref.global "private" constant @__constant_25x10xf64 : memref<25x10xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_10xf64 : memref<10xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_25xf64 : memref<25xf64> = dense<0.000000e+00>
  func @mlir_gmm_opt_full(%arg0: memref<25xf64>, %arg1: memref<25x10xf64>, %arg2: memref<25x10xf64>, %arg3: memref<25x10x10xf64>, %arg4: memref<1000x10xf64>, %arg5: f64, %arg6: i64) -> f64 {
    %cst = arith.constant 0.000000e+00 : f64
    return %cst : f64
  }

  func @__grad_mlir_gmm_opt_full(%arg0: memref<25xf64>, %arg1: memref<25x10xf64>, %arg2: memref<25x10xf64>, %arg3: memref<25x10x10xf64>, %arg4: memref<1000x10xf64>, %arg5: f64, %arg6: i64) -> (memref<25xf64>, memref<25x10xf64>, memref<25x10xf64>, memref<25x10x10xf64>) {
    %c10 = arith.constant 10 : index
    %cst = arith.constant 5.000000e-01 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c25 = arith.constant 25 : index
    %cst_0 = arith.constant 1.000000e+03 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %c24 = arith.constant 24 : index
    %cst_2 = arith.constant 0.000000e+00 : f64
    %c999 = arith.constant 999 : index
    %0 = memref.get_global @__constant_10xf64 : memref<10xf64>
    %1 = memref.get_global @__constant_xf64 : memref<f64>
    %2 = memref.get_global @__constant_25xf64 : memref<25xf64>
    %3 = memref.alloc() : memref<25xf64>
    %4 = memref.alloca() : memref<f64>
    %5 = memref.alloc() : memref<25x10xf64>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<25x10xf64>) outs(%5 : memref<25x10xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %71 = math.exp %arg7 : f64
      linalg.yield %71 : f64
    }
    %6 = memref.alloc() : memref<25xf64>
    linalg.copy(%2, %6) : memref<25xf64>, memref<25xf64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<25x10xf64>) outs(%6 : memref<25xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %71 = arith.addf %arg7, %arg8 : f64
      linalg.yield %71 : f64
    }
    %7 = memref.load %arg0[%c0] : memref<25xf64>
    memref.store %7, %4[] : memref<f64>
    %8 = memref.alloca() : memref<f64>
    linalg.copy(%4, %8) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["reduction"]} ins(%arg0 : memref<25xf64>) outs(%8 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %71 = arith.cmpf ogt, %arg7, %arg8 : f64
      %72 = select %71, %arg7, %arg8 : f64
      linalg.yield %72 : f64
    }
    %9 = memref.load %8[] : memref<f64>
    %10 = memref.alloca() : memref<f64>
    linalg.copy(%1, %10) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["reduction"]} ins(%arg0 : memref<25xf64>) outs(%10 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %71 = arith.subf %arg7, %9 : f64
      %72 = math.exp %71 : f64
      %73 = arith.addf %72, %arg8 : f64
      linalg.yield %73 : f64
    }
    // I think the adjoint begins here?
    %11 = memref.load %10[] : memref<f64>
    %12 = arith.negf %cst_1 : f64
    %13 = memref.alloca() : memref<25xf64>
    linalg.fill(%cst_2, %13) : f64, memref<25xf64> 
    %14 = memref.alloc() : memref<25x10xf64>
    linalg.fill(%cst_2, %14) : f64, memref<25x10xf64> 
    %15 = memref.alloc() : memref<25x10x10xf64>
    linalg.fill(%cst_2, %15) : f64, memref<25x10x10xf64> 
    %16 = memref.alloca() : memref<f64>
    %17 = memref.alloc() : memref<10x10xf64>
    %18 = memref.alloc() : memref<10x10xf64>
    %19 = memref.alloca() : memref<f64>
    %20 = memref.alloc() : memref<10xf64>
    %22:3 = scf.for %arg7 = %c0 to %c25 step %c1 iter_args(%arg8 = %13, %arg9 = %14, %arg10 = %15) -> (memref<25xf64>, memref<25x10xf64>, memref<25x10x10xf64>) {
      %71 = arith.subi %c24, %arg7 : index
      %72 = memref.subview %5[%71, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map2>
      %73 = memref.cast %72 : memref<10xf64, #map2> to memref<10xf64>
      %74 = memref.subview %arg3[%71, 0, 0] [1, 10, 10] [1, 1, 1] : memref<25x10x10xf64> to memref<10x10xf64, #map4>
      %75 = arith.mulf %arg5, %arg5 : f64
      %76 = arith.mulf %75, %cst : f64
      %77 = arith.sitofp %arg6 : i64 to f64
      %78 = arith.negf %cst_1 : f64
      %79 = arith.mulf %78, %77 : f64
      %80 = memref.load %arg8[%71] : memref<25xf64>
      %81 = arith.addf %80, %79 : f64
      memref.store %81, %arg8[%71] : memref<25xf64>
      %82 = arith.mulf %cst_1, %76 : f64
      linalg.fill(%cst_2, %16) : f64, memref<f64> 
      memref.store %82, %16[] : memref<f64>
      scf.for %arg11 = %c0 to %c10 step %c1 {
        scf.for %arg13 = %c0 to %arg11 step %c1 {
          %91 = memref.load %74[%arg11, %arg13] : memref<10x10xf64, #map4>
          %92 = memref.load %16[] : memref<f64>
          %93 = arith.mulf %92, %91 : f64
          %94 = arith.mulf %92, %91 : f64
          %95 = arith.addf %94, %93 : f64
          memref.store %95, %17[%arg11, %arg13] : memref<10x10xf64>
        }
      }
      %84 = memref.subview %arg10[%71, 0, 0] [1, 10, 10] [1, 1, 1] : memref<25x10x10xf64> to memref<10x10xf64, #map4>
      %85 = scf.for %arg11 = %c0 to %c10 step %c1 iter_args(%arg12 = %18) -> (memref<10x10xf64>) {
        %90 = scf.for %arg13 = %c0 to %arg11 step %c1 iter_args(%arg14 = %arg12) -> (memref<10x10xf64>) {
          %91 = memref.load %84[%arg11, %arg13] : memref<10x10xf64, #map4>
          %92 = memref.load %17[%arg11, %arg13] : memref<10x10xf64>
          %93 = arith.addf %91, %92 : f64
          memref.store %93, %arg14[%arg11, %arg13] : memref<10x10xf64>
          scf.yield %arg14 : memref<10x10xf64>
        }
        scf.yield %90 : memref<10x10xf64>
      }
      %86 = memref.subview %arg10[%71, 0, 0] [1, 10, 10] [1, 1, 1] : memref<25x10x10xf64> to memref<10x10xf64, #map4>
      linalg.copy(%85, %84) : memref<10x10xf64>, memref<10x10xf64, #map4> 
      linalg.fill(%cst_2, %19) : f64, memref<f64> 
      memref.store %82, %19[] : memref<f64>
      linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel"]} ins(%73, %19 : memref<10xf64>, memref<f64>) outs(%20 : memref<10xf64>) {
      ^bb0(%arg11: f64, %arg12: f64, %arg13: f64):  // no predecessors
        %90 = arith.mulf %arg12, %arg11 : f64
        %91 = arith.mulf %arg12, %arg11 : f64
        %92 = arith.addf %91, %90 : f64
        linalg.yield %92 : f64
      }
      %87 = memref.subview %arg9[%71, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map2>
      %89 = memref.subview %arg9[%71, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map2>
      linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%20 : memref<10xf64>) outs(%89 : memref<10xf64, #map2>) {
      ^bb0(%arg11: f64, %arg12: f64):  // no predecessors
        %90 = arith.addf %arg11, %arg12 : f64
        linalg.yield %90 : f64
      }
      scf.yield %arg8, %arg9, %arg10 : memref<25xf64>, memref<25x10xf64>, memref<25x10x10xf64>
    }
    %23 = arith.mulf %12, %cst_0 : f64
    %24 = arith.divf %23, %11 : f64
    %25 = memref.alloca() : memref<f64>
    linalg.fill(%cst_2, %25) : f64, memref<f64> 
    memref.store %24, %25[] : memref<f64>
    %26 = memref.alloca() : memref<f64>
    linalg.copy(%1, %26) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map3, #map5, #map5], iterator_types = ["parallel"]} ins(%arg0, %25 : memref<25xf64>, memref<f64>) outs(%26 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %71 = arith.subf %arg7, %9 : f64
      %72 = math.exp %71 : f64
      %73 = arith.mulf %arg8, %72 : f64
      %74 = arith.negf %73 : f64
      %75 = arith.addf %74, %arg9 : f64
      linalg.yield %75 : f64
    }
    %27 = memref.load %26[] : memref<f64>
    %28 = arith.addf %27, %23 : f64
    %29 = memref.alloca() : memref<25xf64>
    linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel"]} ins(%arg0, %25 : memref<25xf64>, memref<f64>) outs(%29 : memref<25xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %71 = arith.subf %arg7, %9 : f64
      %72 = math.exp %71 : f64
      %73 = arith.mulf %arg8, %72 : f64
      linalg.yield %73 : f64
    }
    %30 = memref.alloca() : memref<f64>
    linalg.fill(%cst_2, %30) : f64, memref<f64> 
    memref.store %28, %30[] : memref<f64>
    %31 = memref.alloca() : memref<25xf64>
    linalg.copy(%2, %31) : memref<25xf64>, memref<25xf64> 
    linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel"]} ins(%arg0, %30 : memref<25xf64>, memref<f64>) outs(%31 : memref<25xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %71 = arith.cmpf ogt, %arg7, %arg9 : f64
      %72 = select %71, %arg8, %cst_2 : f64
      linalg.yield %72 : f64
    }
    %32 = memref.alloc() : memref<25xf64>
    linalg.copy(%29, %32) : memref<25xf64>, memref<25xf64> 
    linalg.generic {doc = "Add in place", indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%31 : memref<25xf64>) outs(%32 : memref<25xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %71 = arith.addf %arg7, %arg8 : f64
      linalg.yield %71 : f64
    }
    %33 = memref.load %30[] : memref<f64>
    %34 = memref.load %32[%c0] : memref<25xf64>
    %35 = arith.addf %34, %33 : f64
    memref.store %35, %32[%c0] : memref<25xf64>
    %37 = memref.alloc() : memref<25x10xf64>
    linalg.fill(%cst_2, %37) : f64, memref<25x10xf64> 
    %38 = memref.alloc() : memref<10xf64>
    %39 = memref.alloc() : memref<10xf64>
    %40 = memref.alloc() : memref<10xf64>
    %41 = memref.alloc() : memref<10xf64>
    %42 = memref.alloca() : memref<f64>
    %43 = memref.alloca() : memref<f64>
    %44 = memref.alloca() : memref<f64>
    %45 = memref.alloca() : memref<f64>
    %46 = memref.alloca() : memref<f64>
    %47 = memref.alloc() : memref<25xf64>
    %48 = memref.alloca() : memref<f64>
    %49 = memref.alloc() : memref<25xf64>
    %50 = memref.alloc() : memref<25xf64>
    %51 = memref.alloc() : memref<10xf64>
    %52 = memref.alloc() : memref<10xf64>
    %53 = memref.alloc() : memref<10xf64>
    %54 = memref.alloc() : memref<10xf64>
    %55 = memref.alloca() : memref<f64>
    %56 = memref.alloc() : memref<10xf64>
    %57 = memref.alloc() : memref<10x10xf64>
    %58 = memref.alloc() : memref<10xf64>
    %59 = memref.alloc() : memref<10xf64>
    %60 = memref.alloc() : memref<10xf64>
    %61 = memref.alloc() : memref<10xf64>
    %62 = memref.alloc() : memref<10x10xf64>
    %63 = memref.alloc() : memref<10xf64>
    %64 = memref.alloc() : memref<10xf64>
    %67:3 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %22#0, %arg9 = %22#1, %arg10 = %22#2) -> (memref<25xf64>, memref<25x10xf64>, memref<25x10x10xf64>) {
      %71 = arith.subi %c999, %arg7 : index
      %72 = scf.for %arg14 = %c0 to %c25 step %c1 iter_args(%arg15 = %3) -> (memref<25xf64>) {
        %83 = memref.subview %arg4[%71, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<10xf64, #map2>
        %84 = memref.cast %83 : memref<10xf64, #map2> to memref<10xf64>
        %85 = memref.subview %arg1[%arg14, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map2>
        %86 = memref.cast %85 : memref<10xf64, #map2> to memref<10xf64>
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%84, %86 : memref<10xf64>, memref<10xf64>) outs(%38 : memref<10xf64>) {
        ^bb0(%arg16: f64, %arg17: f64, %arg18: f64):  // no predecessors
          %97 = arith.subf %arg16, %arg17 : f64
          linalg.yield %97 : f64
        }
        %87 = memref.subview %5[%arg14, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map2>
        %88 = memref.cast %87 : memref<10xf64, #map2> to memref<10xf64>
        %89 = memref.subview %arg3[%arg14, 0, 0] [1, 10, 10] [1, 1, 1] : memref<25x10x10xf64> to memref<10x10xf64, #map4>
        %90 = memref.cast %89 : memref<10x10xf64, #map4> to memref<10x10xf64>
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%88, %38 : memref<10xf64>, memref<10xf64>) outs(%39 : memref<10xf64>) {
        ^bb0(%arg16: f64, %arg17: f64, %arg18: f64):  // no predecessors
          %97 = arith.mulf %arg16, %arg17 : f64
          linalg.yield %97 : f64
        }
        linalg.copy(%0, %40) : memref<10xf64>, memref<10xf64> 
        linalg.matvec ins(%90, %38 : memref<10x10xf64>, memref<10xf64>) outs(%40 : memref<10xf64>)
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%39, %40 : memref<10xf64>, memref<10xf64>) outs(%41 : memref<10xf64>) {
        ^bb0(%arg16: f64, %arg17: f64, %arg18: f64):  // no predecessors
          %97 = arith.addf %arg16, %arg17 : f64
          linalg.yield %97 : f64
        }
        linalg.copy(%1, %42) : memref<f64>, memref<f64> 
        linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["reduction"]} ins(%41 : memref<10xf64>) outs(%42 : memref<f64>) {
        ^bb0(%arg16: f64, %arg17: f64):  // no predecessors
          %97 = arith.mulf %arg16, %arg16 : f64
          %98 = arith.addf %97, %arg17 : f64
          linalg.yield %98 : f64
        }
        %91 = memref.load %42[] : memref<f64>
        %92 = arith.mulf %91, %cst : f64
        %93 = memref.load %arg0[%arg14] : memref<25xf64>
        %94 = memref.load %6[%arg14] : memref<25xf64>
        %95 = arith.addf %93, %94 : f64
        %96 = arith.subf %95, %92 : f64
        memref.store %96, %arg15[%arg14] : memref<25xf64>
        scf.yield %arg15 : memref<25xf64>
      }
      %73 = memref.load %72[%c0] : memref<25xf64>
      memref.store %73, %4[] : memref<f64>
      linalg.copy(%4, %43) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["reduction"]} ins(%72 : memref<25xf64>) outs(%43 : memref<f64>) {
      ^bb0(%arg14: f64, %arg15: f64):  // no predecessors
        %83 = arith.cmpf ogt, %arg14, %arg15 : f64
        %84 = select %83, %arg14, %arg15 : f64
        linalg.yield %84 : f64
      }
      %74 = memref.load %43[] : memref<f64>
      linalg.copy(%1, %44) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["reduction"]} ins(%72 : memref<25xf64>) outs(%44 : memref<f64>) {
      ^bb0(%arg14: f64, %arg15: f64):  // no predecessors
        %83 = arith.subf %arg14, %74 : f64
        %84 = math.exp %83 : f64
        %85 = arith.addf %84, %arg15 : f64
        linalg.yield %85 : f64
      }
      %75 = memref.load %44[] : memref<f64>
      %76 = arith.divf %cst_1, %75 : f64
      linalg.fill(%cst_2, %45) : f64, memref<f64> 
      memref.store %76, %45[] : memref<f64>
      linalg.copy(%1, %46) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map3, #map5, #map5], iterator_types = ["parallel"]} ins(%72, %45 : memref<25xf64>, memref<f64>) outs(%46 : memref<f64>) {
      ^bb0(%arg14: f64, %arg15: f64, %arg16: f64):  // no predecessors
        %83 = arith.subf %arg14, %74 : f64
        %84 = math.exp %83 : f64
        %85 = arith.mulf %arg15, %84 : f64
        %86 = arith.negf %85 : f64
        %87 = arith.addf %86, %arg16 : f64
        linalg.yield %87 : f64
      }
      %77 = memref.load %46[] : memref<f64>
      %78 = arith.addf %77, %cst_1 : f64
      linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel"]} ins(%72, %45 : memref<25xf64>, memref<f64>) outs(%47 : memref<25xf64>) {
      ^bb0(%arg14: f64, %arg15: f64, %arg16: f64):  // no predecessors
        %83 = arith.subf %arg14, %74 : f64
        %84 = math.exp %83 : f64
        %85 = arith.mulf %arg15, %84 : f64
        linalg.yield %85 : f64
      }
      linalg.fill(%cst_2, %48) : f64, memref<f64> 
      memref.store %78, %48[] : memref<f64>
      linalg.copy(%2, %49) : memref<25xf64>, memref<25xf64> 
      linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel"]} ins(%72, %48 : memref<25xf64>, memref<f64>) outs(%49 : memref<25xf64>) {
      ^bb0(%arg14: f64, %arg15: f64, %arg16: f64):  // no predecessors
        %83 = arith.cmpf ogt, %arg14, %arg16 : f64
        %84 = select %83, %arg15, %cst_2 : f64
        linalg.yield %84 : f64
      }
      linalg.copy(%47, %50) : memref<25xf64>, memref<25xf64> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%49 : memref<25xf64>) outs(%50 : memref<25xf64>) {
      ^bb0(%arg14: f64, %arg15: f64):  // no predecessors
        %83 = arith.addf %arg14, %arg15 : f64
        linalg.yield %83 : f64
      }
      %79 = memref.load %48[] : memref<f64>
      %80 = memref.load %50[%c0] : memref<25xf64>
      %81 = arith.addf %80, %79 : f64
      memref.store %81, %50[%c0] : memref<25xf64>
      %82:5 = scf.for %arg14 = %c0 to %c25 step %c1 iter_args(%arg15 = %arg8, %arg16 = %arg9, %arg17 = %arg10, %arg19 = %32, %arg20 = %37) -> (memref<25xf64>, memref<25x10xf64>, memref<25x10x10xf64>, memref<25xf64>, memref<25x10xf64>) {
        %83 = arith.subi %c24, %arg14 : index
        %84 = memref.subview %arg4[%71, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<10xf64, #map2>
        %85 = memref.cast %84 : memref<10xf64, #map2> to memref<10xf64>
        %86 = memref.subview %arg1[%83, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map2>
        %87 = memref.cast %86 : memref<10xf64, #map2> to memref<10xf64>
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%85, %87 : memref<10xf64>, memref<10xf64>) outs(%51 : memref<10xf64>) {
        ^bb0(%arg21: f64, %arg22: f64, %arg23: f64):  // no predecessors
          %113 = arith.subf %arg21, %arg22 : f64
          linalg.yield %113 : f64
        }
        %88 = memref.subview %5[%83, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map2>
        %89 = memref.cast %88 : memref<10xf64, #map2> to memref<10xf64>
        %90 = memref.subview %arg3[%83, 0, 0] [1, 10, 10] [1, 1, 1] : memref<25x10x10xf64> to memref<10x10xf64, #map4>
        %91 = memref.cast %90 : memref<10x10xf64, #map4> to memref<10x10xf64>
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%89, %51 : memref<10xf64>, memref<10xf64>) outs(%52 : memref<10xf64>) {
        ^bb0(%arg21: f64, %arg22: f64, %arg23: f64):  // no predecessors
          %113 = arith.mulf %arg21, %arg22 : f64
          linalg.yield %113 : f64
        }
        linalg.copy(%0, %53) : memref<10xf64>, memref<10xf64> 
        linalg.matvec ins(%91, %51 : memref<10x10xf64>, memref<10xf64>) outs(%53 : memref<10xf64>)
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%52, %53 : memref<10xf64>, memref<10xf64>) outs(%54 : memref<10xf64>) {
        ^bb0(%arg21: f64, %arg22: f64, %arg23: f64):  // no predecessors
          %113 = arith.addf %arg21, %arg22 : f64
          linalg.yield %113 : f64
        }
        %92 = memref.load %50[%83] : memref<25xf64>
        %93 = arith.negf %92 : f64
        %94 = memref.load %arg15[%83] : memref<25xf64>
        %95 = arith.addf %94, %92 : f64
        memref.store %95, %arg15[%83] : memref<25xf64>
        %96 = memref.load %arg19[%83] : memref<25xf64>
        %97 = arith.addf %96, %92 : f64
        memref.store %97, %arg19[%83] : memref<25xf64>
        %98 = arith.mulf %93, %cst : f64
        linalg.fill(%cst_2, %55) : f64, memref<f64> 
        memref.store %98, %55[] : memref<f64>
        linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel"]} ins(%54, %55 : memref<10xf64>, memref<f64>) outs(%56 : memref<10xf64>) {
        ^bb0(%arg21: f64, %arg22: f64, %arg23: f64):  // no predecessors
          %113 = arith.mulf %arg22, %arg21 : f64
          %114 = arith.mulf %arg22, %arg21 : f64
          %115 = arith.addf %114, %113 : f64
          linalg.yield %115 : f64
        }
        %99 = scf.for %arg21 = %c0 to %c10 step %c1 iter_args(%arg22 = %57) -> (memref<10x10xf64>) {
          %113 = scf.for %arg23 = %c0 to %arg21 step %c1 iter_args(%arg24 = %arg22) -> (memref<10x10xf64>) {
            %114 = memref.load %56[%arg21] : memref<10xf64>
            %115 = memref.load %51[%arg23] : memref<10xf64>
            %116 = arith.mulf %114, %115 : f64
            memref.store %116, %arg24[%arg21, %arg23] : memref<10x10xf64>
            scf.yield %arg24 : memref<10x10xf64>
          }
          scf.yield %113 : memref<10x10xf64>
        }
        linalg.fill(%cst_2, %58) : f64, memref<10xf64> 
        %100 = scf.for %arg21 = %c0 to %c10 step %c1 iter_args(%arg22 = %58) -> (memref<10xf64>) {
          %113 = scf.for %arg23 = %c0 to %arg21 step %c1 iter_args(%arg24 = %arg22) -> (memref<10xf64>) {
            %114 = memref.load %56[%arg21] : memref<10xf64>
            %115 = memref.load %90[%arg21, %arg23] : memref<10x10xf64, #map4>
            %116 = memref.load %arg24[%arg23] : memref<10xf64>
            %117 = arith.mulf %114, %115 : f64
            %118 = arith.addf %117, %116 : f64
            memref.store %118, %arg24[%arg23] : memref<10xf64>
            scf.yield %arg24 : memref<10xf64>
          }
          scf.yield %113 : memref<10xf64>
        }
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%56, %51 : memref<10xf64>, memref<10xf64>) outs(%59 : memref<10xf64>) {
        ^bb0(%arg21: f64, %arg22: f64, %arg23: f64):  // no predecessors
          %113 = arith.mulf %arg21, %arg22 : f64
          linalg.yield %113 : f64
        }
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%56, %89 : memref<10xf64>, memref<10xf64>) outs(%60 : memref<10xf64>) {
        ^bb0(%arg21: f64, %arg22: f64, %arg23: f64):  // no predecessors
          %113 = arith.mulf %arg21, %arg22 : f64
          linalg.yield %113 : f64
        }
        linalg.copy(%100, %61) : memref<10xf64>, memref<10xf64> 
        linalg.generic {doc = "Add in place", indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%60 : memref<10xf64>) outs(%61 : memref<10xf64>) {
        ^bb0(%arg21: f64, %arg22: f64):  // no predecessors
          %113 = arith.addf %arg21, %arg22 : f64
          linalg.yield %113 : f64
        }
        %101 = memref.subview %arg17[%83, 0, 0] [1, 10, 10] [1, 1, 1] : memref<25x10x10xf64> to memref<10x10xf64, #map4>
        %102 = scf.for %arg21 = %c0 to %c10 step %c1 iter_args(%arg22 = %62) -> (memref<10x10xf64>) {
          %113 = scf.for %arg23 = %c0 to %arg21 step %c1 iter_args(%arg24 = %arg22) -> (memref<10x10xf64>) {
            %114 = memref.load %101[%arg21, %arg23] : memref<10x10xf64, #map4>
            %115 = memref.load %99[%arg21, %arg23] : memref<10x10xf64>
            %116 = arith.addf %114, %115 : f64
            memref.store %116, %arg24[%arg21, %arg23] : memref<10x10xf64>
            scf.yield %arg24 : memref<10x10xf64>
          }
          scf.yield %113 : memref<10x10xf64>
        }
        %103 = memref.subview %arg17[%83, 0, 0] [1, 10, 10] [1, 1, 1] : memref<25x10x10xf64> to memref<10x10xf64, #map4>
        linalg.copy(%102, %103) : memref<10x10xf64>, memref<10x10xf64, #map4> 
        %104 = memref.subview %arg16[%83, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map2>
        %105 = memref.cast %104 : memref<10xf64, #map2> to memref<10xf64>
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%105, %59 : memref<10xf64>, memref<10xf64>) outs(%63 : memref<10xf64>) {
        ^bb0(%arg21: f64, %arg22: f64, %arg23: f64):  // no predecessors
          %113 = arith.addf %arg21, %arg22 : f64
          linalg.yield %113 : f64
        }
        %106 = memref.subview %arg16[%83, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map2>
        linalg.copy(%63, %106) : memref<10xf64>, memref<10xf64, #map2>
        %107 = memref.subview %arg20[%83, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map2>
        linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%61 : memref<10xf64>) outs(%107 : memref<10xf64, #map2>) {
        ^bb0(%arg23: f64, %arg24: f64):  // no predecessors
          %113 = arith.negf %arg23 : f64
          %114 = arith.addf %113, %arg24 : f64
          linalg.yield %114 : f64
        }
        scf.yield %arg15, %arg16, %arg17, %arg19, %arg20 : memref<25xf64>, memref<25x10xf64>, memref<25x10x10xf64>, memref<25xf64>, memref<25x10xf64>
      }
      scf.yield %82#0, %82#1, %82#2 : memref<25xf64>, memref<25x10xf64>, memref<25x10x10xf64>
    }
    %68 = memref.alloc() : memref<25x10xf64>
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %67#0 : memref<25x10xf64>, memref<25xf64>) outs(%68 : memref<25x10xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      linalg.yield %arg8 : f64
    }
    %69 = memref.alloc() : memref<25x10xf64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %67#1 : memref<25x10xf64>, memref<25x10xf64>) outs(%69 : memref<25x10xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %71 = math.exp %arg7 : f64
      %72 = arith.mulf %arg8, %71 : f64
      linalg.yield %72 : f64
    }
    %70 = memref.alloc() : memref<25x10xf64>
    linalg.copy(%68, %70) : memref<25x10xf64>, memref<25x10xf64> 
    linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%69 : memref<25x10xf64>) outs(%70 : memref<25x10xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %71 = arith.addf %arg7, %arg8 : f64
      linalg.yield %71 : f64
    }
    return %32, %37, %70, %67#2 : memref<25xf64>, memref<25x10xf64>, memref<25x10xf64>, memref<25x10x10xf64>
  }
  func @lagrad_gmm_full(%arg0: memref<25xf64>, %arg1: memref<25x10xf64>, %arg2: memref<25x10xf64>, %arg3: memref<25x10x10xf64>, %arg4: memref<1000x10xf64>, %arg5: f64, %arg6: i64) -> (memref<25xf64>, memref<25x10xf64>, memref<25x10xf64>, memref<25x10x10xf64>) {
    %0:4 = call @__grad_mlir_gmm_opt_full(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (memref<25xf64>, memref<25x10xf64>, memref<25x10xf64>, memref<25x10x10xf64>, memref<1000x10xf64>, f64, i64) -> (memref<25xf64>, memref<25x10xf64>, memref<25x10xf64>, memref<25x10x10xf64>)
    return %0#0, %0#1, %0#2, %0#3 : memref<25xf64>, memref<25x10xf64>, memref<25x10xf64>, memref<25x10x10xf64>
  }
}

