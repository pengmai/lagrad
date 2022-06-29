// Made by running lstm.mlir through the bufferization script including buffer deallocation
// Notes: this is using the bufferization strategy that allocates new buffers for each subview, possibly resulting in slower performance.
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
module  {
  memref.global "private" constant @__constant_{{b}}xf64 : memref<{{b}}xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  func private @emsigmoid(%arg0: f64) -> f64 {
    %cst = arith.constant 1.000000e+00 : f64
    %0 = arith.negf %arg0 : f64
    %1 = math.exp %0 : f64
    %2 = arith.addf %cst, %1 : f64
    %3 = arith.divf %cst, %2 : f64
    return %3 : f64
  }
  func @emlogsumexp(%arg0: memref<{{b}}xf64>) -> f64 {
    %cst = arith.constant 2.000000e+00 : f64
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %1 = memref.alloca() : memref<f64>
    linalg.copy(%0, %1) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<{{b}}xf64>) outs(%1 : memref<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):  // no predecessors
      %5 = math.exp %arg1 : f64
      %6 = arith.addf %5, %arg2 : f64
      linalg.yield %6 : f64
    }
    %2 = memref.load %1[] : memref<f64>
    %3 = arith.addf %2, %cst : f64
    %4 = math.log %3 : f64
    return %4 : f64
  }
  // func private @print_memref_f64(memref<*xf64>) attributes {llvm.emit_c_interface}
  func @emlstm_objective(%arg0: memref<{{l}}x2x4x{{b}}xf64>, %arg1: memref<3x{{b}}xf64>, %arg2: memref<{{l}}x2x{{b}}xf64>, %arg3: memref<{{c}}x{{b}}xf64>) -> f64 {
    %cst = arith.constant 0.000000e+00 : f64
    %cl = arith.constant {{l}} : index
    %ccm1 = arith.constant {{c - 1}} : index
    %cb = arith.constant {{b}} : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %1 = memref.alloca() : memref<{{b}}xf64>
    %2 = memref.alloca() : memref<{{b}}xf64>
    %3 = memref.alloca() : memref<{{b}}xf64>
    %4 = memref.alloca() : memref<{{b}}xf64>
    %5 = memref.alloca() : memref<{{b}}xf64>
    %6 = memref.alloca() : memref<{{b}}xf64>
    %7 = memref.alloca() : memref<{{b}}xf64>
    %8 = memref.alloca() : memref<{{b}}xf64>
    %9 = memref.alloca() : memref<{{b}}xf64>
    %10 = memref.alloca() : memref<{{b}}xf64>
    %11 = memref.alloca() : memref<{{b}}xf64>
    %12 = memref.alloca() : memref<{{b}}xf64>
    %13 = memref.alloca() : memref<{{b}}xf64>
    %14 = memref.alloca() : memref<{{b}}xf64>
    %15 = memref.alloca() : memref<{{b}}xf64>
    %16 = memref.alloca() : memref<{{b}}xf64>
    %17 = memref.alloca() : memref<{{b}}xf64>
    %18 = memref.alloca() : memref<{{b}}xf64>
    %19 = memref.alloca() : memref<{{b}}xf64>
    %20 = memref.alloca() : memref<{{b}}xf64>
    %21 = memref.alloca() : memref<{{b}}xf64>
    %22 = memref.alloca() : memref<{{b}}xf64>
    %23 = memref.alloca() : memref<{{b}}xf64>
    %24 = memref.alloca() : memref<f64>
    %25:2 = scf.for %arg4 = %c0 to %ccm1 step %c1 iter_args(%arg5 = %cst, %arg6 = %c0) -> (f64, index) {
      %30 = memref.subview %arg3[%arg4, 0] [1, 14] [1, 1] : memref<{{c}}x{{b}}xf64> to memref<{{b}}xf64, #map2>
      linalg.copy(%30, %1) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
      %31 = memref.subview %arg1[0, 0] [1, 14] [1, 1] : memref<3x{{b}}xf64> to memref<{{b}}xf64, #map2>
      linalg.copy(%31, %2) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
      linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%1, %2 : memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%3 : memref<{{b}}xf64>) {
      ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
        %42 = arith.mulf %arg7, %arg8 : f64
        linalg.yield %42 : f64
      }
      %32 = memref.alloc() : memref<{{b}}xf64>
      %33 = scf.for %arg7 = %c0 to %cl step %c1 iter_args(%arg8 = %3) -> (memref<{{b}}xf64>) {
        %42 = memref.subview %arg0[%arg7, 0, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        linalg.copy(%42, %4) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
        %43 = memref.subview %arg0[%arg7, 1, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        linalg.copy(%43, %5) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
        %44 = memref.subview %arg0[%arg7, 0, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        linalg.copy(%44, %6) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
        %45 = memref.subview %arg0[%arg7, 1, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        linalg.copy(%45, %7) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
        %46 = memref.subview %arg0[%arg7, 0, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        linalg.copy(%46, %8) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
        %47 = memref.subview %arg0[%arg7, 1, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        linalg.copy(%47, %9) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
        %48 = memref.subview %arg0[%arg7, 0, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        linalg.copy(%48, %10) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
        %49 = memref.subview %arg0[%arg7, 1, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        linalg.copy(%49, %11) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
        %50 = memref.subview %arg2[%arg7, 0, 0] [1, 1, 14] [1, 1, 1] : memref<{{l}}x2x{{b}}xf64> to memref<{{b}}xf64, #map2>
        linalg.copy(%50, %12) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
        %51 = memref.subview %arg2[%arg7, 1, 0] [1, 1, 14] [1, 1, 1] : memref<{{l}}x2x{{b}}xf64> to memref<{{b}}xf64, #map2>
        linalg.copy(%51, %13) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
        linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg8, %4, %5 : memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%14 : memref<{{b}}xf64>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
          %54 = arith.mulf %arg9, %arg10 : f64
          %55 = arith.addf %54, %arg11 : f64
          %56 = call @emsigmoid(%55) : (f64) -> f64
          linalg.yield %56 : f64
        }
        linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%12, %6, %7 : memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%15 : memref<{{b}}xf64>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
          %54 = arith.mulf %arg9, %arg10 : f64
          %55 = arith.addf %54, %arg11 : f64
          %56 = call @emsigmoid(%55) : (f64) -> f64
          linalg.yield %56 : f64
        }
        linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg8, %8, %9 : memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%16 : memref<{{b}}xf64>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
          %54 = arith.mulf %arg9, %arg10 : f64
          %55 = arith.addf %54, %arg11 : f64
          %56 = call @emsigmoid(%55) : (f64) -> f64
          linalg.yield %56 : f64
        }
        linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%12, %10, %11 : memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%17 : memref<{{b}}xf64>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
          %54 = arith.mulf %arg9, %arg10 : f64
          %55 = arith.addf %54, %arg11 : f64
          %56 = math.tanh %55 : f64
          linalg.yield %56 : f64
        }
        linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%13, %14, %15, %17 : memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%18 : memref<{{b}}xf64>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64):  // no predecessors
          %54 = arith.mulf %arg9, %arg10 : f64
          %55 = arith.mulf %arg11, %arg12 : f64
          %56 = arith.addf %54, %55 : f64
          linalg.yield %56 : f64
        }
        %52 = memref.subview %arg2[%arg7, 1, 0] [1, 1, 14] [1, 1, 1] : memref<{{l}}x2x{{b}}xf64> to memref<{{b}}xf64, #map2>
        linalg.copy(%18, %52) : memref<{{b}}xf64>, memref<{{b}}xf64, #map2> 
        linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%16, %18 : memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%32 : memref<{{b}}xf64>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
          %54 = math.tanh %arg10 : f64
          %55 = arith.mulf %arg9, %54 : f64
          linalg.yield %55 : f64
        }
        %53 = memref.subview %arg2[%arg7, 0, 0] [1, 1, 14] [1, 1, 1] : memref<{{l}}x2x{{b}}xf64> to memref<{{b}}xf64, #map2>
        linalg.copy(%32, %53) : memref<{{b}}xf64>, memref<{{b}}xf64, #map2> 
        scf.yield %32 : memref<{{b}}xf64>
      }
      %34 = memref.subview %arg1[1, 0] [1, 14] [1, 1] : memref<3x{{b}}xf64> to memref<{{b}}xf64, #map2>
      linalg.copy(%34, %19) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
      %35 = memref.subview %arg1[2, 0] [1, 14] [1, 1] : memref<3x{{b}}xf64> to memref<{{b}}xf64, #map2>
      linalg.copy(%35, %20) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
      linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%33, %19, %20 : memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%21 : memref<{{b}}xf64>) {
      ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
        %42 = arith.mulf %arg7, %arg8 : f64
        %43 = arith.addf %42, %arg9 : f64
        linalg.yield %43 : f64
      }
      memref.dealloc %32 : memref<{{b}}xf64>
      %36 = call @emlogsumexp(%21) : (memref<{{b}}xf64>) -> f64
      linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%21 : memref<{{b}}xf64>) outs(%22 : memref<{{b}}xf64>) {
      ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
        %42 = arith.subf %arg7, %36 : f64
        linalg.yield %42 : f64
      }
      %37 = arith.addi %arg4, %c1 : index
      %38 = memref.subview %arg3[%37, 0] [1, 14] [1, 1] : memref<{{c}}x{{b}}xf64> to memref<{{b}}xf64, #map2>
      linalg.copy(%38, %23) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64> 
      linalg.copy(%0, %24) : memref<f64>, memref<f64> 
      linalg.dot ins(%23, %22 : memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%24 : memref<f64>)
      %39 = memref.load %24[] : memref<f64>
      %40 = arith.addf %39, %arg5 : f64
      %41 = arith.addi %arg6, %cb : index
      scf.yield %40, %41 : f64, index
    }
    %26 = arith.negf %25#0 : f64
    %27 = arith.index_cast %25#1 : index to i64
    %28 = arith.sitofp %27 : i64 to f64
    %29 = arith.divf %26, %28 : f64
    return %29 : f64
  }

  func @enzyme_mlir_lstm(
    %main_params: memref<{{l}}x2x4x{{b}}xf64>,
    %extra_params: memref<3x{{b}}xf64>,
    %state_init: memref<{{l}}x2x{{b}}xf64>,
    %sequence: memref<{{c}}x{{b}}xf64>
  ) -> (memref<{{l}}x2x4x{{b}}xf64>, memref<3x{{b}}xf64>) {
    %zero = arith.constant 0.0 : f64
    %dmain = memref.alloc() : memref<{{l}}x2x4x{{b}}xf64>
    %dextra = memref.alloc() : memref<3x{{b}}xf64>
    %dstate = memref.alloc() : memref<{{l}}x2x{{b}}xf64>
    linalg.fill(%zero, %dmain) : f64, memref<{{l}}x2x4x{{b}}xf64>
    linalg.fill(%zero, %dextra) : f64, memref<3x{{b}}xf64>
    linalg.fill(%zero, %dstate) : f64, memref<{{l}}x2x{{b}}xf64>
    %f = constant @emlstm_objective : (memref<{{l}}x2x4x{{b}}xf64>, memref<3x{{b}}xf64>, memref<{{l}}x2x{{b}}xf64>, memref<{{c}}x{{b}}xf64>) -> f64
    %df = standalone.diff %f {const = [3]} :
      (memref<{{l}}x2x4x{{b}}xf64>, memref<3x{{b}}xf64>, memref<{{l}}x2x{{b}}xf64>, memref<{{c}}x{{b}}xf64>) -> f64,
      (memref<{{l}}x2x4x{{b}}xf64>, memref<{{l}}x2x4x{{b}}xf64>, memref<3x{{b}}xf64>, memref<3x{{b}}xf64>, memref<{{l}}x2x{{b}}xf64>, memref<{{l}}x2x{{b}}xf64>, memref<{{c}}x{{b}}xf64>) -> f64
    call_indirect %df(%main_params, %dmain, %extra_params, %dextra, %state_init, %dstate, %sequence) : (
      memref<{{l}}x2x4x{{b}}xf64>,
      memref<{{l}}x2x4x{{b}}xf64>,
      memref<3x{{b}}xf64>,
      memref<3x{{b}}xf64>,
      memref<{{l}}x2x{{b}}xf64>,
      memref<{{l}}x2x{{b}}xf64>,
      memref<{{c}}x{{b}}xf64>
    ) -> f64
    memref.dealloc %dstate : memref<{{l}}x2x{{b}}xf64>
    return %dmain, %dextra : memref<{{l}}x2x4x{{b}}xf64>, memref<3x{{b}}xf64>
  }
}
