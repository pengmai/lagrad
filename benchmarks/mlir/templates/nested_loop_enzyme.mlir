#map0 = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module  {
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  func @en_nested_loop(%arg0: memref<{{n}}x{{d}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ck = arith.constant {{k}} : index
    %cn = arith.constant {{n}} : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = memref.alloc() : memref<{{k}}xf64>
    linalg.fill(%cst, %0) : f64, memref<{{k}}xf64> 
    %1 = memref.get_global @__constant_xf64 : memref<f64>
    %2 = memref.get_global @__constant_xf64 : memref<f64>
    %3 = memref.alloca() : memref<f64>
    %4 = memref.alloca() : memref<f64>
    %5 = scf.for %arg2 = %c0 to %cn step %c1 iter_args(%arg3 = %cst) -> (f64) {
      %6 = scf.for %arg4 = %c0 to %ck step %c1 iter_args(%arg5 = %0) -> (memref<{{k}}xf64>) {
        %10 = memref.subview %arg0[%arg2, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
        %11 = memref.cast %10 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
        %12 = memref.subview %arg1[%arg4, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
        %13 = memref.cast %12 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
        linalg.copy(%2, %3) : memref<f64>, memref<f64> 
        linalg.dot ins(%11, %13 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%3 : memref<f64>)
        %14 = memref.load %3[] : memref<f64>
        memref.store %14, %arg5[%arg4] : memref<{{k}}xf64>
        scf.yield %arg5 : memref<{{k}}xf64>
      }
      linalg.copy(%1, %4) : memref<f64>, memref<f64> 
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%6 : memref<{{k}}xf64>) outs(%4 : memref<f64>) {
      ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
        %10 = math.exp %arg4 : f64
        %11 = arith.addf %10, %arg5 : f64
        linalg.yield %11 : f64
      }
      %7 = memref.load %4[] : memref<f64>
      %8 = math.log %7 : f64
      %9 = arith.addf %8, %arg3 : f64
      scf.yield %9 : f64
    }
    memref.dealloc %0 : memref<{{k}}xf64>
    return %5 : f64
  }

  func @en_loop(%arg0: memref<{{n}}x{{d}}xf64>) -> f64 {
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %cn = arith.constant {{n}} : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f64
    %1 = scf.for %arg1 = %c0 to %cn step %c1 iter_args(%arg2 = %cst) -> (f64) {
      %2 = memref.subview %arg0[%arg1, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
      %3 = memref.cast %2 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
      %4 = memref.alloc() : memref<f64>
      linalg.copy(%0, %4) : memref<f64>, memref<f64>
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%3 : memref<{{d}}xf64>) outs(%4 : memref<f64>) {
      ^bb0(%arg3: f64, %arg4: f64):  // no predecessors
        %7 = arith.addf %arg3, %arg4 : f64
        linalg.yield %7 : f64
      }
      %5 = memref.load %4[] : memref<f64>
      %6 = arith.addf %5, %arg2 : f64
      scf.yield %6 : f64
    }
    return %1 : f64
  }

  func @en_main_term(%arg0: memref<{{n}}x{{d}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>) -> f64 {
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %cst = arith.constant 0.000000e+00 : f64
    %ck = arith.constant {{k}} : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %1 = memref.alloc() : memref<{{k}}xf64>
    linalg.fill(%cst, %1) : f64, memref<{{k}}xf64>
    %2 = memref.alloca() : memref<f64>
    %sumexp = memref.alloca() : memref<f64>
    memref.store %cst, %sumexp[] : memref<f64>
    %3 = scf.for %arg2 = %c0 to %ck step %c1 iter_args(%arg3 = %1) -> (memref<{{k}}xf64>) {
      %4 = memref.subview %arg0[%arg2, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
      %5 = memref.cast %4 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
      %6 = memref.subview %arg1[%arg2, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
      %7 = memref.cast %6 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
      linalg.copy(%0, %2) : memref<f64>, memref<f64>
      linalg.dot ins(%5, %7 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%2 : memref<f64>)
      %8 = memref.load %2[] : memref<f64>
      memref.store %8, %arg3[%arg2] : memref<{{k}}xf64>
      scf.yield %arg3 : memref<{{k}}xf64>
    }

    linalg.generic
      {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]
      }
      ins(%3 : memref<{{k}}xf64>)
      outs(%sumexp : memref<f64>) {
    ^bb0(%barg0: f64, %barg1: f64):
      %se0 = math.exp %barg0 : f64
      %se1 = arith.addf %se0, %barg1 : f64
      linalg.yield %se1 : f64
    }
    %sumexp_v = memref.load %sumexp[] : memref<f64>
    %lse = math.log %sumexp_v : f64
    return %lse : f64
  }

  func @enzyme_main_term(%A: memref<{{n}}x{{d}}xf64>, %B: memref<{{k}}x{{d}}xf64>) -> memref<{{n}}x{{d}}xf64> {
    %f = constant @en_main_term : (memref<{{n}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) -> f64
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64

    %dA = memref.alloc() : memref<{{n}}x{{d}}xf64>
    linalg.fill(%zero, %dA) : f64, memref<{{n}}x{{d}}xf64>

    // %df = standalone.diff %f {const = [1]} : (memref<{{n}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) -> f64, (memref<{{n}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) -> f64
    // call_indirect %df(%A, %dA, %B) : (memref<{{n}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) -> f64
    return %dA : memref<{{n}}x{{d}}xf64>
  }

  func @enzyme_nested_loop(%A: memref<{{n}}x{{d}}xf64>, %B: memref<{{k}}x{{d}}xf64>) -> memref<{{n}}x{{d}}xf64> {
    %zero = arith.constant 0.0 : f64
    %dA = memref.alloc() : memref<{{n}}x{{d}}xf64>
    linalg.fill(%zero, %dA) : f64, memref<{{n}}x{{d}}xf64>
    %f = constant @en_nested_loop : (memref<{{n}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) -> f64
    %df = standalone.diff %f {const = [1]} : (memref<{{n}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) -> f64, (memref<{{n}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) -> f64
    call_indirect %df(%A, %dA, %B) : (memref<{{n}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>) -> f64
    return %dA : memref<{{n}}x{{d}}xf64>
  }

  func @enzyme_loop(%A: memref<{{n}}x{{d}}xf64>) -> memref<{{n}}x{{d}}xf64> {
    %zero = arith.constant 0.0 : f64
    %dA = memref.alloc() : memref<{{n}}x{{d}}xf64>
    linalg.fill(%zero, %dA) : f64, memref<{{n}}x{{d}}xf64>
    // %f = constant @en_loop : (memref<{{n}}x{{d}}xf64>) -> f64
    // %df = standalone.diff %f : (memref<{{n}}x{{d}}xf64>) -> f64, (memref<{{n}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>) -> f64
    // call_indirect %df(%A, %dA) : (memref<{{n}}x{{d}}xf64>, memref<{{n}}x{{d}}xf64>) -> f64
    return %dA : memref<{{n}}x{{d}}xf64>
  }
}
