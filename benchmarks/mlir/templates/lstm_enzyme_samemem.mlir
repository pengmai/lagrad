// Made by running lstm.mlir through the bufferization script including buffer deallocation,
// then modifying the result to use the same memory allocation pattern as the C version.
// This somehow gets very good performance at the cost of high memory usage.
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#view = affine_map<(d0)[s0] -> (d0 + s0)>
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
  func @emlstm_objective(%arg0: memref<{{l}}x2x4x{{b}}xf64>, %arg1: memref<3x{{b}}xf64>, %arg2: memref<{{l}}x2x{{b}}xf64>, %arg3: memref<{{c}}x{{b}}xf64>) -> f64 {
    %cst = arith.constant 0.000000e+00 : f64
    %c2 = arith.constant 2 : index
    %ccm1 = arith.constant {{c - 1}} : index
    %cb = arith.constant {{b}} : index
    %cl = arith.constant {{l}} : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %ypred = memref.alloc() : memref<{{b}}xf64>
    %ynorm = memref.alloc() : memref<{{b}}xf64>
    // %6 = memref.alloca() : memref<{{b}}xf64>
    // %7 = memref.alloca() : memref<{{b}}xf64>
    // %8 = memref.alloca() : memref<{{b}}xf64>
    %9 = memref.alloca() : memref<f64>
    %10:2 = scf.for %arg4 = %c0 to %ccm1 step %c1 iter_args(%arg5 = %cst, %arg6 = %c0) -> (f64, index) {
      %15 = memref.subview %arg3[%arg4, 0] [1, {{b}}] [1, 1] : memref<{{c}}x{{b}}xf64> to memref<{{b}}xf64, #map2>
      %16 = memref.cast %15 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
      %17 = memref.subview %arg1[0, 0] [1, {{b}}] [1, 1] : memref<3x{{b}}xf64> to memref<{{b}}xf64, #map2>
      %18 = memref.cast %17 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
      linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%16, %18 : memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%ypred : memref<{{b}}xf64>) {
      ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
        %32 = arith.mulf %arg7, %arg8 : f64
        linalg.yield %32 : f64
      }
      %19 = memref.alloca() : memref<{{b}}xf64>
      %20 = scf.for %arg7 = %c0 to %cl step %c1 iter_args(%arg8 = %ypred) -> (memref<{{b}}xf64>) {
        %gates = memref.alloc() : memref<4x{{b}}xf64>
        %forget  = memref.subview %gates[0, 0] [1, 14] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
        %ingate  = memref.subview %gates[1, 0] [1, 14] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
        %outgate = memref.subview %gates[2, 0] [1, 14] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
        %change  = memref.subview %gates[3, 0] [1, 14] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
        %32 = memref.subview %arg0[%arg7, 0, 0, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        %33 = memref.cast %32 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
        %34 = memref.subview %arg0[%arg7, 1, 0, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        %35 = memref.cast %34 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
        %36 = memref.subview %arg0[%arg7, 0, 1, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        %37 = memref.cast %36 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
        %38 = memref.subview %arg0[%arg7, 1, 1, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        %39 = memref.cast %38 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
        %40 = memref.subview %arg0[%arg7, 0, 2, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        %41 = memref.cast %40 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
        %42 = memref.subview %arg0[%arg7, 1, 2, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        %43 = memref.cast %42 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
        %44 = memref.subview %arg0[%arg7, 0, 3, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        %45 = memref.cast %44 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
        %46 = memref.subview %arg0[%arg7, 1, 3, 0] [1, 1, 1, {{b}}] [1, 1, 1, 1] : memref<{{l}}x2x4x{{b}}xf64> to memref<{{b}}xf64, #map2>
        %47 = memref.cast %46 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
        %48 = memref.subview %arg2[%arg7, 0, 0] [1, 1, {{b}}] [1, 1, 1] : memref<{{l}}x2x{{b}}xf64> to memref<{{b}}xf64, #map2>
        %49 = memref.cast %48 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
        %50 = memref.subview %arg2[%arg7, 1, 0] [1, 1, {{b}}] [1, 1, 1] : memref<{{l}}x2x{{b}}xf64> to memref<{{b}}xf64, #map2>
        %51 = memref.cast %50 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
        linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg8, %33, %35 : memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%forget : memref<{{b}}xf64, #view>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
          %54 = arith.mulf %arg9, %arg10 : f64
          %55 = arith.addf %54, %arg11 : f64
          %56 = call @emsigmoid(%55) : (f64) -> f64
          linalg.yield %56 : f64
        }
        linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%49, %37, %39 : memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%ingate : memref<{{b}}xf64, #view>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
          %54 = arith.mulf %arg9, %arg10 : f64
          %55 = arith.addf %54, %arg11 : f64
          %56 = call @emsigmoid(%55) : (f64) -> f64
          linalg.yield %56 : f64
        }
        linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg8, %41, %43 : memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%outgate : memref<{{b}}xf64, #view>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
          %54 = arith.mulf %arg9, %arg10 : f64
          %55 = arith.addf %54, %arg11 : f64
          %56 = call @emsigmoid(%55) : (f64) -> f64
          linalg.yield %56 : f64
        }
        linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%49, %45, %47 : memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%change : memref<{{b}}xf64, #view>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
          %54 = arith.mulf %arg9, %arg10 : f64
          %55 = arith.addf %54, %arg11 : f64
          %56 = math.tanh %55 : f64
          linalg.yield %56 : f64
        }
        linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%51, %forget, %ingate, %change : memref<{{b}}xf64>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>) outs(%50 : memref<{{b}}xf64, #map2>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64):  // no predecessors
          %54 = arith.mulf %arg9, %arg10 : f64
          %55 = arith.mulf %arg11, %arg12 : f64
          %56 = arith.addf %54, %55 : f64
          linalg.yield %56 : f64
        }
        // %52 = memref.subview %arg2[%arg7, 1, 0] [1, 1, {{b}}] [1, 1, 1] : memref<{{l}}x2x{{b}}xf64> to memref<{{b}}xf64, #map2>
        // linalg.copy(%6, %52) : memref<{{b}}xf64>, memref<{{b}}xf64, #map2> 
        linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%outgate, %50 : memref<{{b}}xf64, #view>, memref<{{b}}xf64, #map2>) outs(%48 : memref<{{b}}xf64, #map2>) {
        ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
          %54 = math.tanh %arg10 : f64
          %55 = arith.mulf %arg9, %54 : f64
          linalg.yield %55 : f64
        }
        linalg.copy(%48, %19) : memref<{{b}}xf64, #map2>, memref<{{b}}xf64>
        // %53 = memref.subview %arg2[%arg7, 0, 0] [1, 1, {{b}}] [1, 1, 1] : memref<{{l}}x2x{{b}}xf64> to memref<{{b}}xf64, #map2>
        // linalg.copy(%19, %53) : memref<{{b}}xf64>, memref<{{b}}xf64, #map2>
        memref.dealloc %gates : memref<4x{{b}}xf64>
        scf.yield %19 : memref<{{b}}xf64>
      }
      %21 = memref.subview %arg1[1, 0] [1, {{b}}] [1, 1] : memref<3x{{b}}xf64> to memref<{{b}}xf64, #map2>
      %22 = memref.cast %21 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
      %23 = memref.subview %arg1[2, 0] [1, {{b}}] [1, 1] : memref<3x{{b}}xf64> to memref<{{b}}xf64, #map2>
      %24 = memref.cast %23 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
      linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%20, %22, %24 : memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%ypred : memref<{{b}}xf64>) {
      ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
        %32 = arith.mulf %arg7, %arg8 : f64
        %33 = arith.addf %32, %arg9 : f64
        linalg.yield %33 : f64
      }
      %25 = call @emlogsumexp(%ypred) : (memref<{{b}}xf64>) -> f64
      linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%ypred : memref<{{b}}xf64>) outs(%ynorm : memref<{{b}}xf64>) {
      ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
        %32 = arith.subf %arg7, %25 : f64
        linalg.yield %32 : f64
      }
      %26 = arith.addi %arg4, %c1 : index
      %27 = memref.subview %arg3[%26, 0] [1, {{b}}] [1, 1] : memref<{{c}}x{{b}}xf64> to memref<{{b}}xf64, #map2>
      %28 = memref.cast %27 : memref<{{b}}xf64, #map2> to memref<{{b}}xf64>
      linalg.copy(%0, %9) : memref<f64>, memref<f64> 
      linalg.dot ins(%28, %ynorm : memref<{{b}}xf64>, memref<{{b}}xf64>) outs(%9 : memref<f64>)
      %29 = memref.load %9[] : memref<f64>
      %30 = arith.addf %29, %arg5 : f64
      %31 = arith.addi %arg6, %cb : index
      scf.yield %30, %31 : f64, index
    }
    %11 = arith.negf %10#0 : f64
    %12 = arith.index_cast %10#1 : index to i64
    %13 = arith.sitofp %12 : i64 to f64
    %14 = arith.divf %11, %13 : f64
    memref.dealloc %ypred : memref<{{b}}xf64>
    memref.dealloc %ynorm : memref<{{b}}xf64>
    return %14 : f64
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
