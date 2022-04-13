#map0 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map1 = affine_map<(d0) -> (d0)>
module  {
  func @manual_cache_loop(%arg0: memref<{{n}}x{{d}}xf64>) -> memref<{{n}}x{{d}}xf64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant {{n}} : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %c511 = arith.constant {{n - 1}} : index
    %c510 = arith.constant {{n - 2}} : index
    %0 = memref.alloc() : memref<{{n}}x{{d}}xf64>
    %1 = memref.alloca() : memref<{{d}}xf64>
    scf.for %arg1 = %c0 to %c512 step %c1 {
      %6 = arith.cmpi eq, %arg1, %c0 : index
      %7 = memref.subview %arg0[%arg1, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
      %8 = memref.cast %7 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
      %9 = scf.if %6 -> (memref<{{d}}xf64>) {
        scf.yield %8 : memref<{{d}}xf64>
      } else {
        %11 = arith.subi %arg1, %c1 : index
        %12 = memref.subview %0[%11, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
        %13 = memref.cast %12 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%13, %8 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%1 : memref<{{d}}xf64>) {
        ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
          %14 = arith.mulf %arg2, %arg3 : f64
          linalg.yield %14 : f64
        }
        scf.yield %1 : memref<{{d}}xf64>
      }
      %10 = memref.subview %0[%arg1, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
      linalg.copy(%9, %10) : memref<{{d}}xf64>, memref<{{d}}xf64, #map0> 
    }
    %2 = memref.alloc() : memref<{{n}}x{{d}}xf64>
    linalg.fill(%cst_0, %2) : f64, memref<{{n}}x{{d}}xf64> 
    %3 = memref.alloc() : memref<{{n}}x{{d}}xf64>
    linalg.fill(%cst, %3) : f64, memref<{{n}}x{{d}}xf64> 
    %4 = memref.alloca() : memref<{{d}}xf64>

    scf.for %arg1 = %c0 to %c512 step %c1 {
      %6 = arith.subi %c511, %arg1 : index
      %7 = memref.subview %arg0[%6, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
      %8 = memref.cast %7 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
      %9 = memref.subview %2[%6, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
      %10 = memref.cast %9 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
      %11 = arith.cmpi eq, %6, %c0 : index
      %12 = scf.if %11 -> (memref<{{d}}xf64>) {
        scf.yield %10 : memref<{{d}}xf64>
      } else {
        %16 = arith.subi %c510, %arg1 : index
        %17 = memref.subview %0[%16, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
        %18 = memref.cast %17 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
        %19 = memref.subview %2[%16, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
        %20 = memref.cast %19 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%18, %10 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%4 : memref<{{d}}xf64>) {
        ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
          %25 = arith.mulf %arg2, %arg3 : f64
          linalg.yield %25 : f64
        }

        %22 = memref.subview %2[%16, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
        linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel"]} ins(%20, %8, %10 : memref<{{d}}xf64>, memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%22 : memref<{{d}}xf64, #map0>) {
        ^bb0(%arg2: f64, %arg3: f64, %arg4: f64, %arg5: f64):  // no predecessors
          %25 = arith.mulf %arg3, %arg4 : f64
          %26 = arith.addf %arg2, %25 : f64
          linalg.yield %26 : f64
        }

        // %24 = memref.subview %2[%6, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
        // linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%10, %8 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%24 : memref<{{d}}xf64, #map0>) {
        // ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
        //   %25 = arith.mulf %arg3, %arg2 : f64
        //   %26 = arith.addf %arg2, %25 : f64
        //   linalg.yield %26 : f64
        // }
        scf.yield %4 : memref<{{d}}xf64>
      }
      %13 = memref.subview %3[%6, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
      %14 = memref.cast %13 : memref<{{d}}xf64, #map0> to memref<{{d}}xf64>
      %15 = memref.subview %3[%6, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map0>
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%14, %12 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%15 : memref<{{d}}xf64, #map0>) {
      ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
        %16 = arith.addf %arg2, %arg3 : f64
        linalg.yield %16 : f64
      }
    }
    // memref.dealloc %1 : memref<{{d}}xf64>
    memref.dealloc %2 : memref<{{n}}x{{d}}xf64>
    memref.dealloc %0 : memref<{{n}}x{{d}}xf64>
    // memref.dealloc %4 : memref<{{d}}xf64>
    return %3 : memref<{{n}}x{{d}}xf64>
  }
}

