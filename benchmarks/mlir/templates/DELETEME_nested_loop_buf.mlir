#map0 = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module  {
  memref.global "private" constant @__constant_10xf64 : memref<10xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_25xf64 : memref<25xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  func @lg_nested_loop(%arg0: memref<1000x10xf64>, %arg1: memref<25x10xf64>) -> f64 {
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %cst = arith.constant 0.000000e+00 : f64
    %c1000 = arith.constant 1000 : index
    %c25 = arith.constant 25 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %1 = memref.alloca() : memref<25xf64>
    linalg.fill(%cst, %1) : f64, memref<25xf64> 
    %2 = memref.alloca() : memref<f64>
    %3 = memref.alloca() : memref<f64>
    %4 = scf.for %arg2 = %c0 to %c1000 step %c1 iter_args(%arg3 = %cst) -> (f64) {
      %5 = memref.subview %arg0[%arg2, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<10xf64, #map0>
      %6 = memref.cast %5 : memref<10xf64, #map0> to memref<10xf64>
      %7 = scf.for %arg4 = %c0 to %c25 step %c1 iter_args(%arg5 = %1) -> (memref<25xf64>) {
        %11 = memref.subview %arg1[%arg4, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map0>
        %12 = memref.cast %11 : memref<10xf64, #map0> to memref<10xf64>
        linalg.copy(%0, %2) : memref<f64>, memref<f64> 
        linalg.dot ins(%6, %12 : memref<10xf64>, memref<10xf64>) outs(%2 : memref<f64>)
        %13 = memref.load %2[] : memref<f64>
        memref.store %13, %arg5[%arg4] : memref<25xf64>
        scf.yield %arg5 : memref<25xf64>
      }
      linalg.copy(%0, %3) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%7 : memref<25xf64>) outs(%3 : memref<f64>) {
      ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
        %11 = math.exp %arg4 : f64
        %12 = arith.addf %11, %arg5 : f64
        linalg.yield %12 : f64
      }
      %8 = memref.load %3[] : memref<f64>
      %9 = math.log %8 : f64
      %10 = arith.addf %9, %arg3 : f64
      scf.yield %10 : f64
    }
    return %4 : f64
  }
  func @__grad_lg_nested_loop(%arg0: memref<1000x10xf64>, %arg1: memref<25x10xf64>) -> memref<1000x10xf64> {
    %0 = memref.get_global @__constant_25xf64 : memref<25xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %1 = memref.get_global @__constant_xf64 : memref<f64>
    %c1000 = arith.constant 1000 : index
    %c25 = arith.constant 25 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %2 = memref.alloca() : memref<25xf64>
    linalg.fill(%cst, %2) : f64, memref<25xf64> 
    %3 = memref.alloc() : memref<1000x10xf64>
    linalg.fill(%cst, %3) : f64, memref<1000x10xf64> 
    %4 = memref.alloca() : memref<f64>
    %5 = memref.alloca() : memref<f64>
    %6 = memref.alloca() : memref<f64>
    %7 = memref.alloca() : memref<25xf64>
    %8 = memref.alloca() : memref<10xf64>
    %9 = memref.alloca() : memref<f64>
    %10 = memref.alloca() : memref<10xf64>
    %22 = memref.alloca() : memref<10xf64>
    %11 = scf.for %arg2 = %c0 to %c1000 step %c1 iter_args(%arg3 = %3) -> (memref<1000x10xf64>) {
      %12 = memref.subview %arg0[%arg2, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<10xf64, #map0>
      %13 = memref.cast %12 : memref<10xf64, #map0> to memref<10xf64>
      %14 = scf.for %arg4 = %c0 to %c25 step %c1 iter_args(%arg5 = %2) -> (memref<25xf64>) {
        %19 = memref.subview %arg1[%arg4, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map0>
        %20 = memref.cast %19 : memref<10xf64, #map0> to memref<10xf64>
        linalg.copy(%1, %4) : memref<f64>, memref<f64> 
        linalg.dot ins(%13, %20 : memref<10xf64>, memref<10xf64>) outs(%4 : memref<f64>)
        %21 = memref.load %4[] : memref<f64>
        memref.store %21, %arg5[%arg4] : memref<25xf64>
        scf.yield %arg5 : memref<25xf64>
      }
      linalg.copy(%1, %5) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%14 : memref<25xf64>) outs(%5 : memref<f64>) {
      ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
        %19 = math.exp %arg4 : f64
        %20 = arith.addf %19, %arg5 : f64
        linalg.yield %20 : f64
      }
      %15 = memref.load %5[] : memref<f64>
      %16 = arith.divf %cst_0, %15 : f64
      memref.store %16, %6[] : memref<f64>
      linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%14, %6 : memref<25xf64>, memref<f64>) outs(%7 : memref<25xf64>) {
      ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):  // no predecessors
        %19 = math.exp %arg4 : f64
        %20 = arith.mulf %arg5, %19 : f64
        linalg.yield %20 : f64
      }
      linalg.fill(%cst, %8) : f64, memref<10xf64> 
      %17 = scf.for %arg4 = %c0 to %c25 step %c1 iter_args(%arg5 = %8) -> (memref<10xf64>) {
        %19 = memref.subview %arg1[%arg4, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map0>
        %20 = memref.cast %19 : memref<10xf64, #map0> to memref<10xf64>
        %21 = memref.load %7[%arg4] : memref<25xf64>
        memref.store %21, %9[] : memref<f64>
        // linalg.generic {doc = "Copy and scalar multiplication", indexing_maps = [#map2, #map1, #map1, #map1], iterator_types = ["parallel"], library_call = "sdot_grad_first"} ins(%9, %20, %arg5 : memref<f64>, memref<10xf64>, memref<10xf64>) outs(%10 : memref<10xf64>) {
        // ^bb0(%arg6: f64, %arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
        //   %23 = arith.mulf %arg6, %arg7 : f64
        //   // %24 = arith.addf %23, %arg8 : f64
        //   linalg.yield %23 : f64
        // }
        linalg.generic {doc = "Copy and scalar multiplication", indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel"], library_call = "sdot_grad_first"} ins(%9, %20 : memref<f64>, memref<10xf64>) outs(%10 : memref<10xf64>) {
        ^bb0(%arg6: f64, %arg7: f64, %arg8: f64):  // no predecessors
          %23 = arith.mulf %arg6, %arg7 : f64
          linalg.yield %23 : f64
        }
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg5, %10 : memref<10xf64>, memref<10xf64>) outs(%22 : memref<10xf64>) {
        ^bb0(%arg6: f64, %arg7: f64, %arg8: f64):  // no predecessors
          %23 = arith.addf %arg6, %arg7 : f64
          linalg.yield %23 : f64
        }
        scf.yield %22 : memref<10xf64>
      }
      %18 = memref.subview %arg3[%arg2, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<10xf64, #map0>
      linalg.copy(%17, %18) : memref<10xf64>, memref<10xf64, #map0> 
      scf.yield %arg3 : memref<1000x10xf64>
    }
    return %11 : memref<1000x10xf64>
  }
  func @lg_main_term(%arg0: memref<1000x10xf64>, %arg1: memref<25x10xf64>) -> f64 {
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %cst = arith.constant 0.000000e+00 : f64
    %c25 = arith.constant 25 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %1 = memref.alloca() : memref<25xf64>
    linalg.fill(%cst, %1) : f64, memref<25xf64> 
    %2 = memref.alloca() : memref<f64>
    %3 = scf.for %arg2 = %c0 to %c25 step %c1 iter_args(%arg3 = %1) -> (memref<25xf64>) {
      %7 = memref.subview %arg0[%arg2, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<10xf64, #map0>
      %8 = memref.cast %7 : memref<10xf64, #map0> to memref<10xf64>
      %9 = memref.subview %arg1[%arg2, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map0>
      %10 = memref.cast %9 : memref<10xf64, #map0> to memref<10xf64>
      linalg.copy(%0, %2) : memref<f64>, memref<f64> 
      linalg.dot ins(%8, %10 : memref<10xf64>, memref<10xf64>) outs(%2 : memref<f64>)
      %11 = memref.load %2[] : memref<f64>
      memref.store %11, %arg3[%arg2] : memref<25xf64>
      scf.yield %arg3 : memref<25xf64>
    }
    %4 = memref.alloca() : memref<f64>
    linalg.copy(%0, %4) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%3 : memref<25xf64>) outs(%4 : memref<f64>) {
    ^bb0(%arg2: f64, %arg3: f64):  // no predecessors
      %7 = math.exp %arg2 : f64
      %8 = arith.addf %7, %arg3 : f64
      linalg.yield %8 : f64
    }
    %5 = memref.load %4[] : memref<f64>
    %6 = math.log %5 : f64
    return %6 : f64
  }
  func @__grad_lg_main_term(%arg0: memref<1000x10xf64>, %arg1: memref<25x10xf64>) -> memref<1000x10xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = memref.get_global @__constant_25xf64 : memref<25xf64>
    %cst_0 = arith.constant 1.000000e+00 : f64
    %1 = memref.get_global @__constant_xf64 : memref<f64>
    %c25 = arith.constant 25 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %2 = memref.alloca() : memref<25xf64>
    linalg.fill(%cst, %2) : f64, memref<25xf64> 
    %3 = memref.alloca() : memref<f64>
    %4 = scf.for %arg2 = %c0 to %c25 step %c1 iter_args(%arg3 = %2) -> (memref<25xf64>) {
      %14 = memref.subview %arg0[%arg2, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<10xf64, #map0>
      %15 = memref.cast %14 : memref<10xf64, #map0> to memref<10xf64>
      %16 = memref.subview %arg1[%arg2, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map0>
      %17 = memref.cast %16 : memref<10xf64, #map0> to memref<10xf64>
      linalg.copy(%1, %3) : memref<f64>, memref<f64> 
      linalg.dot ins(%15, %17 : memref<10xf64>, memref<10xf64>) outs(%3 : memref<f64>)
      %18 = memref.load %3[] : memref<f64>
      memref.store %18, %arg3[%arg2] : memref<25xf64>
      scf.yield %arg3 : memref<25xf64>
    }
    %5 = memref.alloca() : memref<f64>
    linalg.copy(%1, %5) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%4 : memref<25xf64>) outs(%5 : memref<f64>) {
    ^bb0(%arg2: f64, %arg3: f64):  // no predecessors
      %14 = math.exp %arg2 : f64
      %15 = arith.addf %14, %arg3 : f64
      linalg.yield %15 : f64
    }
    %6 = memref.load %5[] : memref<f64>
    %7 = arith.divf %cst_0, %6 : f64
    %8 = memref.alloca() : memref<f64>
    memref.store %7, %8[] : memref<f64>
    %9 = memref.alloca() : memref<25xf64>
    linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%4, %8 : memref<25xf64>, memref<f64>) outs(%9 : memref<25xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
      %14 = math.exp %arg2 : f64
      %15 = arith.mulf %arg3, %14 : f64
      linalg.yield %15 : f64
    }
    %10 = memref.alloc() : memref<1000x10xf64>
    linalg.fill(%cst, %10) : f64, memref<1000x10xf64> 
    %11 = memref.alloca() : memref<f64>
    %12 = memref.alloca() : memref<10xf64>
    %13 = scf.for %arg2 = %c0 to %c25 step %c1 iter_args(%arg3 = %10) -> (memref<1000x10xf64>) {
      %14 = memref.subview %arg0[%arg2, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<10xf64, #map0>
      %15 = memref.cast %14 : memref<10xf64, #map0> to memref<10xf64>
      %16 = memref.subview %arg1[%arg2, 0] [1, 10] [1, 1] : memref<25x10xf64> to memref<10xf64, #map0>
      %17 = memref.cast %16 : memref<10xf64, #map0> to memref<10xf64>
      %18 = memref.load %9[%arg2] : memref<25xf64>
      memref.store %18, %11[] : memref<f64>
      linalg.generic {doc = "Copy and scalar multiplication", indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel"], library_call = "sdot_grad_first"} ins(%11, %17 : memref<f64>, memref<10xf64>) outs(%12 : memref<10xf64>) {
      ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):  // no predecessors
        %20 = arith.mulf %arg4, %arg5 : f64
        linalg.yield %20 : f64
      }
      %19 = memref.subview %arg3[%arg2, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<10xf64, #map0>
      linalg.copy(%12, %19) : memref<10xf64>, memref<10xf64, #map0> 
      scf.yield %arg3 : memref<1000x10xf64>
    }
    return %13 : memref<1000x10xf64>
  }
  func @lagrad_main_term(%arg0: memref<1000x10xf64>, %arg1: memref<25x10xf64>) -> memref<1000x10xf64> {
    %0 = call @__grad_lg_main_term(%arg0, %arg1) : (memref<1000x10xf64>, memref<25x10xf64>) -> memref<1000x10xf64>
    return %0 : memref<1000x10xf64>
  }
  func @lg_loop(%arg0: memref<1000x10xf64>) -> f64 {
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f64
    %1 = memref.alloca() : memref<f64>
    %2 = scf.for %arg1 = %c0 to %c1000 step %c1 iter_args(%arg2 = %cst) -> (f64) {
      %3 = memref.subview %arg0[%arg1, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<10xf64, #map0>
      %4 = memref.cast %3 : memref<10xf64, #map0> to memref<10xf64>
      linalg.copy(%0, %1) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%4 : memref<10xf64>) outs(%1 : memref<f64>) {
      ^bb0(%arg3: f64, %arg4: f64):  // no predecessors
        %7 = arith.addf %arg3, %arg4 : f64
        linalg.yield %7 : f64
      }
      %5 = memref.load %1[] : memref<f64>
      %6 = arith.addf %5, %arg2 : f64
      scf.yield %6 : f64
    }
    return %2 : f64
  }
  func @__grad_lg_loop(%arg0: memref<1000x10xf64>) -> memref<1000x10xf64> {
    %0 = memref.get_global @__constant_10xf64 : memref<10xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %1 = memref.alloc() : memref<1000x10xf64>
    linalg.fill(%cst, %1) : f64, memref<1000x10xf64> 
    %2 = memref.alloca() : memref<f64>
    %3 = memref.alloca() : memref<10xf64>
    %4 = scf.for %arg1 = %c0 to %c1000 step %c1 iter_args(%arg2 = %1) -> (memref<1000x10xf64>) {
      %5 = memref.subview %arg0[%arg1, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<10xf64, #map0>
      %6 = memref.cast %5 : memref<10xf64, #map0> to memref<10xf64>
      memref.store %cst_0, %2[] : memref<f64>
      linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%6, %2 : memref<10xf64>, memref<f64>) outs(%3 : memref<10xf64>) {
      ^bb0(%arg3: f64, %arg4: f64, %arg5: f64):  // no predecessors
        linalg.yield %arg4 : f64
      }
      %7 = memref.subview %arg2[%arg1, 0] [1, 10] [1, 1] : memref<1000x10xf64> to memref<10xf64, #map0>
      linalg.copy(%3, %7) : memref<10xf64>, memref<10xf64, #map0> 
      scf.yield %arg2 : memref<1000x10xf64>
    }
    return %4 : memref<1000x10xf64>
  }
  func @lagrad_nested_loop(%arg0: memref<1000x10xf64>, %arg1: memref<25x10xf64>) -> memref<1000x10xf64> {
    %0 = call @__grad_lg_nested_loop(%arg0, %arg1) : (memref<1000x10xf64>, memref<25x10xf64>) -> memref<1000x10xf64>
    return %0 : memref<1000x10xf64>
  }
  func @lagrad_loop(%arg0: memref<1000x10xf64>) -> memref<1000x10xf64> {
    %0 = call @__grad_lg_loop(%arg0) : (memref<1000x10xf64>) -> memref<1000x10xf64>
    return %0 : memref<1000x10xf64>
  }
}

