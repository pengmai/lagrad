#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>
#map4 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map5 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s0 + d1 * s2 + s1)>
module  {
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  func @mlir_gmm_opt_full(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>, %arg2: memref<{{k}}x{{d}}xf64>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64>, %arg4: memref<{{n}}x{{d}}xf64>, %arg5: f64, %arg6: i64) -> f64 {
    %cst = arith.constant 0.000000e+00 : f64
    return %cst : f64
  }

  func private @print_memref_f64(memref<*xf64>) -> () attributes {llvm.emit_c_interface}
  func private @dmatvec(memref<{{d}}x{{d}}xf64>, memref<{{d}}xf64>, memref<{{d}}xf64>) attributes {llvm.emit_c_interface}

  func @__grad_mlir_gmm_opt_full(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>, %arg2: memref<{{k}}x{{d}}xf64>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64>, %arg4: memref<{{n}}x{{d}}xf64>, %arg5: f64, %arg6: i64) -> (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>) {
    %c999 = arith.constant {{n - 1}} : index
    %cst = arith.constant 0.000000e+00 : f64
    %c24 = arith.constant {{k - 1}} : index
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+03 : f64
    %c25 = arith.constant {{k}} : index
    %c1000 = arith.constant {{n}} : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 5.000000e-01 : f64
    %c10 = arith.constant {{d}} : index
    %1 = memref.get_global @__constant_xf64 : memref<f64>
    %3 = memref.alloc() : memref<{{k}}xf64>
    %4 = memref.alloca() : memref<f64>
    %5 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%5 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %64 = math.exp %arg7 : f64
      linalg.yield %64 : f64
    }
    %6 = memref.alloc() : memref<{{k}}xf64>
    linalg.fill(%cst, %6) : f64, memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%6 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %64 = arith.addf %arg7, %arg8 : f64
      linalg.yield %64 : f64
    }
    %7 = memref.load %arg0[%c0] : memref<{{k}}xf64>
    memref.store %7, %4[] : memref<f64>
    %8 = memref.alloca() : memref<f64>
    linalg.copy(%4, %8) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction"]} ins(%arg0 : memref<{{k}}xf64>) outs(%8 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %64 = arith.cmpf ogt, %arg7, %arg8 : f64
      %65 = select %64, %arg7, %arg8 : f64
      linalg.yield %65 : f64
    }
    %9 = memref.load %8[] : memref<f64>
    %10 = memref.alloca() : memref<f64>
    linalg.copy(%1, %10) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction"]} ins(%arg0 : memref<{{k}}xf64>) outs(%10 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %64 = arith.subf %arg7, %9 : f64
      %65 = math.exp %64 : f64
      %66 = arith.addf %65, %arg8 : f64
      linalg.yield %66 : f64
    }
    %11 = memref.load %10[] : memref<f64>
    %12 = arith.negf %cst_0 : f64
    %13 = memref.alloc() : memref<{{k}}xf64>
    linalg.fill(%cst, %13) : f64, memref<{{k}}xf64> 
    %14 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.fill(%cst, %14) : f64, memref<{{k}}x{{d}}xf64> 
    %15 = memref.alloc() : memref<{{k}}x{{d}}x{{d}}xf64>
    linalg.fill(%cst, %15) : f64, memref<{{k}}x{{d}}x{{d}}xf64> 
    %16 = memref.alloca() : memref<f64>
    %17 = memref.alloc() : memref<{{d}}x{{d}}xf64>
    %19 = memref.alloca() : memref<f64>
    %20 = memref.alloc() : memref<{{d}}xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %64 = arith.subi %c24, %arg7 : index
      %65 = memref.subview %5[%64, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map4>
      %66 = memref.cast %65 : memref<{{d}}xf64, #map4> to memref<{{d}}xf64>
      %67 = memref.subview %arg3[%64, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : memref<{{k}}x{{d}}x{{d}}xf64> to memref<{{d}}x{{d}}xf64, #map5>
      %68 = arith.mulf %arg5, %arg5 : f64
      %69 = arith.mulf %68, %cst_2 : f64
      %70 = arith.sitofp %arg6 : i64 to f64
      %71 = arith.negf %cst_0 : f64
      %72 = arith.mulf %71, %70 : f64
      %73 = memref.load %13[%64] : memref<{{k}}xf64>
      %74 = arith.addf %73, %72 : f64
      memref.store %74, %13[%64] : memref<{{k}}xf64>
      %75 = arith.mulf %cst_0, %69 : f64
      linalg.fill(%cst, %16) : f64, memref<f64> 
      memref.store %75, %16[] : memref<f64>
      scf.for %arg8 = %c0 to %c10 step %c1 {
        scf.for %arg9 = %c0 to %arg8 step %c1 {
          %78 = memref.load %67[%arg8, %arg9] : memref<{{d}}x{{d}}xf64, #map5>
          %79 = memref.load %16[] : memref<f64>
          %80 = arith.mulf %79, %78 : f64
          %81 = arith.mulf %79, %78 : f64
          %82 = arith.addf %81, %80 : f64
          memref.store %82, %17[%arg8, %arg9] : memref<{{d}}x{{d}}xf64>
        }
      }
      %76 = memref.subview %15[%64, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : memref<{{k}}x{{d}}x{{d}}xf64> to memref<{{d}}x{{d}}xf64, #map5>
      scf.for %arg8 = %c0 to %c10 step %c1 {
        scf.for %arg9 = %c0 to %arg8 step %c1 {
          %78 = memref.load %76[%arg8, %arg9] : memref<{{d}}x{{d}}xf64, #map5>
          %79 = memref.load %17[%arg8, %arg9] : memref<{{d}}x{{d}}xf64>
          %80 = arith.addf %78, %79 : f64
          memref.store %80, %76[%arg8, %arg9] : memref<{{d}}x{{d}}xf64, #map5>
        }
      }
      linalg.fill(%cst, %19) : f64, memref<f64> 
      memref.store %75, %19[] : memref<f64>
      linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%66, %19 : memref<{{d}}xf64>, memref<f64>) outs(%20 : memref<{{d}}xf64>) {
      ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
        %78 = arith.mulf %arg9, %arg8 : f64
        %79 = arith.mulf %arg9, %arg8 : f64
        %80 = arith.addf %79, %78 : f64
        linalg.yield %80 : f64
      }
      %77 = memref.subview %14[%64, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map4>
      linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%20 : memref<{{d}}xf64>) outs(%77 : memref<{{d}}xf64, #map4>) {
      ^bb0(%arg8: f64, %arg9: f64):  // no predecessors
        %78 = arith.addf %arg8, %arg9 : f64
        linalg.yield %78 : f64
      }
    }
    %21 = arith.mulf %12, %cst_1 : f64
    %22 = arith.divf %21, %11 : f64
    %23 = memref.alloca() : memref<f64>
    linalg.fill(%cst, %23) : f64, memref<f64> 
    memref.store %22, %23[] : memref<f64>
    %24 = memref.alloca() : memref<f64>
    linalg.copy(%1, %24) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["parallel"]} ins(%arg0, %23 : memref<{{k}}xf64>, memref<f64>) outs(%24 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %64 = arith.subf %arg7, %9 : f64
      %65 = math.exp %64 : f64
      %66 = arith.mulf %arg8, %65 : f64
      %67 = arith.negf %66 : f64
      %68 = arith.addf %67, %arg9 : f64
      linalg.yield %68 : f64
    }
    %25 = memref.load %24[] : memref<f64>
    %26 = arith.addf %25, %21 : f64
    %27 = memref.alloc() : memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%arg0, %23 : memref<{{k}}xf64>, memref<f64>) outs(%27 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %64 = arith.subf %arg7, %9 : f64
      %65 = math.exp %64 : f64
      %66 = arith.mulf %arg8, %65 : f64
      linalg.yield %66 : f64
    }
    %28 = memref.alloca() : memref<f64>
    linalg.fill(%cst, %28) : f64, memref<f64> 
    memref.store %26, %28[] : memref<f64>
    %29 = memref.alloc() : memref<{{k}}xf64>
    linalg.fill(%cst, %29) : f64, memref<{{k}}xf64>
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%arg0, %28 : memref<{{k}}xf64>, memref<f64>) outs(%29 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %64 = arith.cmpf ogt, %arg7, %arg9 : f64
      %65 = select %64, %arg8, %cst : f64
      linalg.yield %65 : f64
    }
    %30 = memref.alloc() : memref<{{k}}xf64>
    linalg.copy(%27, %30) : memref<{{k}}xf64>, memref<{{k}}xf64> 
    linalg.generic {doc = "Add in place", indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%29 : memref<{{k}}xf64>) outs(%30 : memref<{{k}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %64 = arith.addf %arg7, %arg8 : f64
      linalg.yield %64 : f64
    }
    %31 = memref.load %28[] : memref<f64>
    %32 = memref.load %30[%c0] : memref<{{k}}xf64>
    %33 = arith.addf %32, %31 : f64
    memref.store %33, %30[%c0] : memref<{{k}}xf64>
    %34 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    linalg.fill(%cst, %34) : f64, memref<{{k}}x{{d}}xf64> 
    %35 = memref.alloc() : memref<{{d}}xf64>
    %37 = memref.alloc() : memref<{{d}}xf64> // don't think I can get rid of this
    %39 = memref.alloca() : memref<f64>
    %40 = memref.alloca() : memref<f64>
    %41 = memref.alloca() : memref<f64>
    %42 = memref.alloca() : memref<f64>
    %43 = memref.alloca() : memref<f64>
    %44 = memref.alloc() : memref<{{k}}xf64>
    %45 = memref.alloca() : memref<f64>
    %46 = memref.alloc() : memref<{{k}}xf64>
    %48 = memref.alloc() : memref<{{d}}xf64>
    %49 = memref.alloc() : memref<{{d}}xf64>
    %50 = memref.alloc() : memref<{{d}}xf64>
    %52 = memref.alloca() : memref<f64>
    %54 = memref.alloc() : memref<{{d}}x{{d}}xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      %64 = arith.subi %c999, %arg7 : index
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %74 = memref.subview %arg4[%64, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map4>
        %75 = memref.cast %74 : memref<{{d}}xf64, #map4> to memref<{{d}}xf64>
        %76 = memref.subview %arg1[%arg8, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map4>
        %77 = memref.cast %76 : memref<{{d}}xf64, #map4> to memref<{{d}}xf64>
        linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%75, %77 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%35 : memref<{{d}}xf64>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
          %88 = arith.subf %arg9, %arg10 : f64
          linalg.yield %88 : f64
        }
        %78 = memref.subview %5[%arg8, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map4>
        %79 = memref.cast %78 : memref<{{d}}xf64, #map4> to memref<{{d}}xf64>
        %80 = memref.subview %arg3[%arg8, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : memref<{{k}}x{{d}}x{{d}}xf64> to memref<{{d}}x{{d}}xf64, #map5>
        %81 = memref.cast %80 : memref<{{d}}x{{d}}xf64, #map5> to memref<{{d}}x{{d}}xf64>
        linalg.fill(%cst, %37) : f64, memref<{{d}}xf64>
        scf.for %iv = %c0 to %c10 step %c1 {
          scf.for %jv = %c0 to %iv step %c1 {
            %b0 = memref.load %81[%iv, %jv] : memref<{{d}}x{{d}}xf64>
            %b1 = memref.load %35[%jv] : memref<{{d}}xf64>
            %b2 = memref.load %37[%iv] : memref<{{d}}xf64>
            %b3 = arith.mulf %b0, %b1 : f64
            %b4 = arith.addf %b3, %b2 : f64
            memref.store %b4, %37[%iv] : memref<{{d}}xf64>
          }
        }
        // call @dmatvec(%81, %35, %37) : (memref<{{d}}x{{d}}xf64>, memref<{{d}}xf64>, memref<{{d}}xf64>) -> ()
        // linalg.matvec ins(%81, %35 : memref<{{d}}x{{d}}xf64>, memref<{{d}}xf64>) outs(%37 : memref<{{d}}xf64>)
        // %U = memref.cast %37 : memref<{{d}}xf64> to memref<*xf64>
        // call @print_memref_f64(%U) : (memref<*xf64>) -> ()
        linalg.fill(%cst, %39) : f64, memref<f64>
        linalg.generic {indexing_maps = [#map2, #map2, #map2, #map3], iterator_types = ["parallel"]} ins(%78, %35, %37 : memref<{{d}}xf64, #map4>, memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%39 : memref<f64>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
          %88 = arith.mulf %arg9, %arg10 : f64
          %89 = arith.addf %88, %arg11 : f64
          %90 = arith.mulf %89, %89 : f64
          %91 = arith.addf %90, %arg12 : f64
          linalg.yield %91 : f64
        }
        %82 = memref.load %39[] : memref<f64>
        %83 = arith.mulf %82, %cst_2 : f64
        %84 = memref.load %arg0[%arg8] : memref<{{k}}xf64>
        %85 = memref.load %6[%arg8] : memref<{{k}}xf64>
        %86 = arith.addf %84, %85 : f64
        %87 = arith.subf %86, %83 : f64
        memref.store %87, %3[%arg8] : memref<{{k}}xf64>
      }
      %65 = memref.load %3[%c0] : memref<{{k}}xf64>
      memref.store %65, %4[] : memref<f64>
      linalg.copy(%4, %40) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction"]} ins(%3 : memref<{{k}}xf64>) outs(%40 : memref<f64>) {
      ^bb0(%arg8: f64, %arg9: f64):  // no predecessors
        %74 = arith.cmpf ogt, %arg8, %arg9 : f64
        %75 = select %74, %arg8, %arg9 : f64
        linalg.yield %75 : f64
      }
      %66 = memref.load %40[] : memref<f64>
      linalg.copy(%1, %41) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction"]} ins(%3 : memref<{{k}}xf64>) outs(%41 : memref<f64>) {
      ^bb0(%arg8: f64, %arg9: f64):  // no predecessors
        %74 = arith.subf %arg8, %66 : f64
        %75 = math.exp %74 : f64
        %76 = arith.addf %75, %arg9 : f64
        linalg.yield %76 : f64
      }
      %67 = memref.load %41[] : memref<f64>
      %68 = arith.divf %cst_0, %67 : f64
      linalg.fill(%cst, %42) : f64, memref<f64> 
      memref.store %68, %42[] : memref<f64>
      linalg.copy(%1, %43) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["parallel"]} ins(%3, %42 : memref<{{k}}xf64>, memref<f64>) outs(%43 : memref<f64>) {
      ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
        %74 = arith.subf %arg8, %66 : f64
        %75 = math.exp %74 : f64
        %76 = arith.mulf %arg9, %75 : f64
        %77 = arith.negf %76 : f64
        %78 = arith.addf %77, %arg10 : f64
        linalg.yield %78 : f64
      }
      %69 = memref.load %43[] : memref<f64>
      %70 = arith.addf %69, %cst_0 : f64
      linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%3, %42 : memref<{{k}}xf64>, memref<f64>) outs(%44 : memref<{{k}}xf64>) {
      ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
        %74 = arith.subf %arg8, %66 : f64
        %75 = math.exp %74 : f64
        %76 = arith.mulf %arg9, %75 : f64
        linalg.yield %76 : f64
      }
      linalg.fill(%cst, %45) : f64, memref<f64> 
      memref.store %70, %45[] : memref<f64>
      linalg.fill(%cst, %46) : f64, memref<{{k}}xf64>
      linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%3, %45 : memref<{{k}}xf64>, memref<f64>) outs(%46 : memref<{{k}}xf64>) {
      ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
        %74 = arith.cmpf ogt, %arg8, %arg10 : f64
        %75 = select %74, %arg9, %cst : f64
        linalg.yield %75 : f64
      }
      // linalg.copy(%44, %47) : memref<{{k}}xf64>, memref<{{k}}xf64> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%46 : memref<{{k}}xf64>) outs(%44 : memref<{{k}}xf64>) {
      ^bb0(%arg8: f64, %arg9: f64):  // no predecessors
        %74 = arith.addf %arg8, %arg9 : f64
        linalg.yield %74 : f64
      }
      %71 = memref.load %45[] : memref<f64>
      %72 = memref.load %44[%c0] : memref<{{k}}xf64>
      %73 = arith.addf %72, %71 : f64
      memref.store %73, %44[%c0] : memref<{{k}}xf64>
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %74 = arith.subi %c24, %arg8 : index
        %75 = memref.subview %arg4[%64, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map4>
        %76 = memref.cast %75 : memref<{{d}}xf64, #map4> to memref<{{d}}xf64>
        %77 = memref.subview %arg1[%74, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map4>
        %78 = memref.cast %77 : memref<{{d}}xf64, #map4> to memref<{{d}}xf64>
        linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%76, %78 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%48 : memref<{{d}}xf64>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
          %96 = arith.subf %arg9, %arg10 : f64
          linalg.yield %96 : f64
        }
        %79 = memref.subview %5[%74, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map4>
        %80 = memref.cast %79 : memref<{{d}}xf64, #map4> to memref<{{d}}xf64>
        %81 = memref.subview %arg3[%74, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : memref<{{k}}x{{d}}x{{d}}xf64> to memref<{{d}}x{{d}}xf64, #map5>
        %82 = memref.cast %81 : memref<{{d}}x{{d}}xf64, #map5> to memref<{{d}}x{{d}}xf64>
        linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%80, %48 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%49 : memref<{{d}}xf64>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
          %96 = arith.mulf %arg9, %arg10 : f64
          linalg.yield %96 : f64
        }
        linalg.fill(%cst, %50) : f64, memref<{{d}}xf64>
        scf.for %iv = %c0 to %c10 step %c1 {
          scf.for %jv = %c0 to %iv step %c1 {
            %b0 = memref.load %82[%iv, %jv] : memref<{{d}}x{{d}}xf64>
            %b1 = memref.load %48[%jv] : memref<{{d}}xf64>
            %b2 = memref.load %50[%iv] : memref<{{d}}xf64>
            %b3 = arith.mulf %b0, %b1 : f64
            %b4 = arith.addf %b3, %b2 : f64
            memref.store %b4, %50[%iv] : memref<{{d}}xf64>
          }
        }
        // linalg.matvec ins(%82, %48 : memref<{{d}}x{{d}}xf64>, memref<{{d}}xf64>) outs(%50 : memref<{{d}}xf64>)
        linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%50 : memref<{{d}}xf64>) outs(%49 : memref<{{d}}xf64>) {
        ^bb0(%arg9: f64, %arg10: f64):  // no predecessors
          %96 = arith.addf %arg9, %arg10 : f64
          linalg.yield %96 : f64
        }
        %83 = memref.load %44[%74] : memref<{{k}}xf64>
        %84 = arith.negf %83 : f64
        %85 = memref.load %13[%74] : memref<{{k}}xf64>
        %86 = arith.addf %85, %83 : f64
        memref.store %86, %13[%74] : memref<{{k}}xf64>
        %87 = memref.load %30[%74] : memref<{{k}}xf64>
        %88 = arith.addf %87, %83 : f64
        memref.store %88, %30[%74] : memref<{{k}}xf64>
        %89 = arith.mulf %84, %cst_2 : f64
        linalg.fill(%cst, %52) : f64, memref<f64> 
        memref.store %89, %52[] : memref<f64>
        linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel"]} ins(%52 : memref<f64>) outs(%49 : memref<{{d}}xf64>) {
        ^bb0(%arg9: f64, %arg10: f64):  // no predecessors
          %96 = arith.mulf %arg10, %arg9 : f64
          %97 = arith.mulf %arg10, %arg9 : f64
          %98 = arith.addf %97, %96 : f64
          linalg.yield %98 : f64
        }
        scf.for %arg9 = %c0 to %c10 step %c1 {
          scf.for %arg10 = %c0 to %arg9 step %c1 {
            %96 = memref.load %49[%arg9] : memref<{{d}}xf64>
            %97 = memref.load %48[%arg10] : memref<{{d}}xf64>
            %98 = arith.mulf %96, %97 : f64
            memref.store %98, %54[%arg9, %arg10] : memref<{{d}}x{{d}}xf64>
          }
        }
        linalg.fill(%cst, %50) : f64, memref<{{d}}xf64>
        scf.for %arg9 = %c0 to %c10 step %c1 {
          scf.for %arg10 = %c0 to %arg9 step %c1 {
            %96 = memref.load %49[%arg9] : memref<{{d}}xf64>
            %97 = memref.load %81[%arg9, %arg10] : memref<{{d}}x{{d}}xf64, #map5>
            %98 = memref.load %50[%arg10] : memref<{{d}}xf64>
            %99 = arith.mulf %96, %97 : f64
            %100 = arith.addf %99, %98 : f64
            memref.store %100, %50[%arg10] : memref<{{d}}xf64>
          }
        }
        linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%49, %80 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%50 : memref<{{d}}xf64>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
          %96 = arith.mulf %arg9, %arg10 : f64
          %97 = arith.addf %96, %arg11 : f64
          linalg.yield %97 : f64
        }
        %90 = memref.subview %15[%74, 0, 0] [1, {{d}}, {{d}}] [1, 1, 1] : memref<{{k}}x{{d}}x{{d}}xf64> to memref<{{d}}x{{d}}xf64, #map5>
        scf.for %arg9 = %c0 to %c10 step %c1 {
          scf.for %arg10 = %c0 to %arg9 step %c1 {
            %96 = memref.load %90[%arg9, %arg10] : memref<{{d}}x{{d}}xf64, #map5>
            %97 = memref.load %54[%arg9, %arg10] : memref<{{d}}x{{d}}xf64>
            %98 = arith.addf %96, %97 : f64
            memref.store %98, %90[%arg9, %arg10] : memref<{{d}}x{{d}}xf64, #map5>
          }
        }
        %92 = memref.subview %14[%74, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map4>
        linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%49, %48 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%92 : memref<{{d}}xf64, #map4>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
          %96 = arith.mulf %arg9, %arg10 : f64
          %97 = arith.addf %96, %arg11 : f64
          linalg.yield %97 : f64
        }
        %95 = memref.subview %34[%74, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map4>
        linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%50 : memref<{{d}}xf64>) outs(%95 : memref<{{d}}xf64, #map4>) {
        ^bb0(%arg9: f64, %arg10: f64):  // no predecessors
          %96 = arith.negf %arg9 : f64
          %97 = arith.addf %96, %arg10 : f64
          linalg.yield %97 : f64
        }
      }
    }
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %13 : memref<{{k}}x{{d}}xf64>, memref<{{k}}xf64>) outs(%14 : memref<{{k}}x{{d}}xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %64 = math.exp %arg7 : f64
      %65 = arith.mulf %arg9, %64 : f64
      %66 = arith.addf %65, %arg8 : f64
      linalg.yield %66 : f64
    }
    return %30, %34, %14, %15 : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>
  }
  func @lagrad_gmm_full(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>, %arg2: memref<{{k}}x{{d}}xf64>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64>, %arg4: memref<{{n}}x{{d}}xf64>, %arg5: f64, %arg6: i64) -> (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>) {
    %0:4 = call @__grad_mlir_gmm_opt_full(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>, f64, i64) -> (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>)
    return %0#0, %0#1, %0#2, %0#3 : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>
  }
}

