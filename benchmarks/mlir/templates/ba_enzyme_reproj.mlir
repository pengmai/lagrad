// This was automatically produced by taking the LAGrad version, commenting out the .grad functions,
// and running it through bufferization before modifying mlir_compute_reproj_error to write to an out buffer.

#map0 = affine_map<(d0) -> (d0 + 3)>
#map1 = affine_map<(d0) -> (d0 + 9)>
#map2 = affine_map<(d0) -> (d0 + 7)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0) -> ()>
#map5 = affine_map<(d0) -> ((d0 + 1) mod 3)>
#map6 = affine_map<(d0) -> ((d0 + 2) mod 3)>
module  {
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_3xf64 : memref<3xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_2xf64 : memref<2xf64> = dense<0.000000e+00>
  // func private @print_memref_f64(memref<*xf64>) attributes {llvm.emit_c_interface}
  func @emlir_compute_reproj_error(%arg0: memref<11xf64>, %arg1: memref<3xf64>, %arg2: memref<f64>, %arg3: memref<2xf64>, %out: memref<2xf64>) -> f64 {
    %0 = memref.get_global @__constant_2xf64 : memref<2xf64>
    %1 = memref.alloca() : memref<3xf64>
    %2 = memref.subview %arg0[0] [3] [1] : memref<11xf64> to memref<3xf64>
    linalg.copy(%2, %1) : memref<3xf64>, memref<3xf64> 
    %3 = memref.alloca() : memref<3xf64>
    %4 = memref.subview %arg0[3] [3] [1] : memref<11xf64> to memref<3xf64, #map0>
    linalg.copy(%4, %3) : memref<3xf64, #map0>, memref<3xf64> 
    %5 = memref.alloca() : memref<2xf64>
    %6 = memref.subview %arg0[9] [2] [1] : memref<11xf64> to memref<2xf64, #map1>
    linalg.copy(%6, %5) : memref<2xf64, #map1>, memref<2xf64> 
    %c6 = arith.constant 6 : index
    %7 = memref.load %arg0[%c6] : memref<11xf64>
    %8 = memref.alloca() : memref<2xf64>
    %9 = memref.subview %arg0[7] [2] [1] : memref<11xf64> to memref<2xf64, #map2>
    linalg.copy(%9, %8) : memref<2xf64, #map2>, memref<2xf64> 
    %10 = memref.alloca() : memref<3xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%arg1, %3 : memref<3xf64>, memref<3xf64>) outs(%10 : memref<3xf64>) {
    ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):  // no predecessors
      %42 = arith.subf %arg4, %arg5 : f64
      linalg.yield %42 : f64
    }
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %11 = memref.get_global @__constant_3xf64 : memref<3xf64>
    %12 = memref.get_global @__constant_xf64 : memref<f64>
    %13 = memref.get_global @__constant_xf64 : memref<f64>
    %14 = memref.get_global @__constant_3xf64 : memref<3xf64>
    %15 = memref.get_global @__constant_3xf64 : memref<3xf64>
    %16 = memref.alloca() : memref<f64>
    linalg.copy(%13, %16) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["reduction"]} ins(%1 : memref<3xf64>) outs(%16 : memref<f64>) {
    ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
      %42 = arith.mulf %arg4, %arg4 : f64
      %43 = arith.addf %42, %arg5 : f64
      linalg.yield %43 : f64
    }
    %17 = memref.load %16[] : memref<f64>
    %18 = arith.cmpf one, %17, %cst : f64
    %19 = memref.alloca() : memref<3xf64>
    %20 = memref.alloca() : memref<3xf64>
    %21 = scf.if %18 -> (memref<3xf64>) {
      %42 = math.sqrt %17 : f64
      %43 = math.cos %42 : f64
      %44 = math.sin %42 : f64
      %45 = arith.divf %cst_0, %42 : f64
      %46 = memref.alloca() : memref<3xf64>
      linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%1 : memref<3xf64>) outs(%46 : memref<3xf64>) {
      ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
        %56 = arith.mulf %arg4, %45 : f64
        linalg.yield %56 : f64
      }
      %47 = memref.alloca() : memref<3xf64>
      linalg.generic {indexing_maps = [#map5, #map6, #map6, #map5, #map3], iterator_types = ["parallel"]} ins(%46, %10, %46, %10 : memref<3xf64>, memref<3xf64>, memref<3xf64>, memref<3xf64>) outs(%47 : memref<3xf64>) {
      ^bb0(%arg4: f64, %arg5: f64, %arg6: f64, %arg7: f64, %arg8: f64):  // no predecessors
        %56 = arith.mulf %arg4, %arg5 : f64
        %57 = arith.mulf %arg6, %arg7 : f64
        %58 = arith.subf %56, %57 : f64
        linalg.yield %58 : f64
      }
      %48 = memref.alloca() : memref<f64>
      linalg.copy(%12, %48) : memref<f64>, memref<f64> 
      linalg.dot ins(%46, %10 : memref<3xf64>, memref<3xf64>) outs(%48 : memref<f64>)
      %49 = memref.load %48[] : memref<f64>
      %50 = arith.subf %cst_0, %43 : f64
      %51 = arith.mulf %49, %50 : f64
      %52 = memref.alloca() : memref<3xf64>
      linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%10 : memref<3xf64>) outs(%52 : memref<3xf64>) {
      ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
        %56 = arith.mulf %arg4, %43 : f64
        linalg.yield %56 : f64
      }
      %53 = memref.alloca() : memref<3xf64>
      linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%47 : memref<3xf64>) outs(%53 : memref<3xf64>) {
      ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
        %56 = arith.mulf %arg4, %44 : f64
        linalg.yield %56 : f64
      }
      %54 = memref.alloca() : memref<3xf64>
      linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%46 : memref<3xf64>) outs(%54 : memref<3xf64>) {
      ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
        %56 = arith.mulf %arg4, %51 : f64
        linalg.yield %56 : f64
      }
      %55 = memref.alloca() : memref<3xf64>
      linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%52, %53 : memref<3xf64>, memref<3xf64>) outs(%55 : memref<3xf64>) {
      ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):  // no predecessors
        %56 = arith.addf %arg4, %arg5 : f64
        linalg.yield %56 : f64
      }
      linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%55, %54 : memref<3xf64>, memref<3xf64>) outs(%19 : memref<3xf64>) {
      ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):  // no predecessors
        %56 = arith.addf %arg4, %arg5 : f64
        linalg.yield %56 : f64
      }
      scf.yield %19 : memref<3xf64>
    } else {
      %42 = memref.alloca() : memref<3xf64>
      linalg.generic {indexing_maps = [#map5, #map6, #map6, #map5, #map3], iterator_types = ["parallel"]} ins(%1, %10, %1, %10 : memref<3xf64>, memref<3xf64>, memref<3xf64>, memref<3xf64>) outs(%42 : memref<3xf64>) {
      ^bb0(%arg4: f64, %arg5: f64, %arg6: f64, %arg7: f64, %arg8: f64):  // no predecessors
        %43 = arith.mulf %arg4, %arg5 : f64
        %44 = arith.mulf %arg6, %arg7 : f64
        %45 = arith.subf %43, %44 : f64
        linalg.yield %45 : f64
      }
      linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%10, %42 : memref<3xf64>, memref<3xf64>) outs(%20 : memref<3xf64>) {
      ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):  // no predecessors
        %43 = arith.addf %arg4, %arg5 : f64
        linalg.yield %43 : f64
      }
      scf.yield %20 : memref<3xf64>
    }
    %22 = memref.alloca() : memref<2xf64>
    %23 = memref.subview %21[0] [2] [1] : memref<3xf64> to memref<2xf64>
    linalg.copy(%23, %22) : memref<2xf64>, memref<2xf64> 
    %c2 = arith.constant 2 : index
    %24 = memref.load %21[%c2] : memref<3xf64>
    %25 = memref.get_global @__constant_2xf64 : memref<2xf64>
    %26 = memref.alloca() : memref<2xf64>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%22 : memref<2xf64>) outs(%26 : memref<2xf64>) {
    ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
      %42 = arith.divf %arg4, %24 : f64
      linalg.yield %42 : f64
    }
    %27 = memref.get_global @__constant_xf64 : memref<f64>
    %28 = memref.alloca() : memref<f64>
    linalg.copy(%27, %28) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["reduction"]} ins(%26 : memref<2xf64>) outs(%28 : memref<f64>) {
    ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
      %42 = arith.mulf %arg4, %arg4 : f64
      %43 = arith.addf %42, %arg5 : f64
      linalg.yield %43 : f64
    }
    %29 = memref.load %28[] : memref<f64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant 1.000000e+00 : f64
    %30 = memref.load %5[%c0] : memref<2xf64>
    %31 = memref.load %5[%c1] : memref<2xf64>
    %32 = arith.mulf %30, %29 : f64
    %33 = arith.mulf %31, %29 : f64
    %34 = arith.mulf %33, %29 : f64
    %35 = arith.addf %32, %34 : f64
    %36 = arith.addf %cst_1, %35 : f64
    %37 = memref.get_global @__constant_2xf64 : memref<2xf64>
    %38 = memref.alloca() : memref<2xf64>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%26 : memref<2xf64>) outs(%38 : memref<2xf64>) {
    ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
      %42 = arith.mulf %arg4, %36 : f64
      linalg.yield %42 : f64
    }
    %39 = memref.alloca() : memref<2xf64>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%38 : memref<2xf64>) outs(%39 : memref<2xf64>) {
    ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
      %42 = arith.mulf %arg4, %7 : f64
      linalg.yield %42 : f64
    }
    %40 = memref.alloca() : memref<2xf64>
    linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%39, %8 : memref<2xf64>, memref<2xf64>) outs(%40 : memref<2xf64>) {
    ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):  // no predecessors
      %42 = arith.addf %arg4, %arg5 : f64
      linalg.yield %42 : f64
    }
    %41 = memref.alloca() : memref<2xf64>
    %w = memref.load %arg2[] : memref<f64>
    linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%40, %arg3 : memref<2xf64>, memref<2xf64>) outs(%41 : memref<2xf64>) {
    ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):  // no predecessors
      %42 = arith.subf %arg4, %arg5 : f64
      %43 = arith.mulf %w, %42 : f64
      linalg.yield %43 : f64
    }
    linalg.copy(%41, %out) : memref<2xf64>, memref<2xf64>
    %ret = arith.constant 0.0 : f64
    return %ret : f64
  }

  func @enzyme_compute_reproj_error(
    %cam: memref<11xf64>,
    %X: memref<3xf64>,
    %w: f64,
    %feat: memref<2xf64>,
    %g: memref<2xf64>
  ) -> (memref<11xf64>, memref<3xf64>, f64) {
    %out = memref.alloca() : memref<2xf64>
    %zero = arith.constant 0.0 : f64
    linalg.fill(%zero, %out) : f64, memref<2xf64>
    %wm = memref.alloca() : memref<f64>
    memref.store %w, %wm[] : memref<f64>
    %dcam = memref.alloc() : memref<11xf64>
    linalg.fill(%zero, %dcam) : f64, memref<11xf64>
    %dX = memref.alloc() : memref<3xf64>
    linalg.fill(%zero, %dX) : f64, memref<3xf64>
    %dwm = memref.alloca() : memref<f64>
    memref.store %zero, %dwm[] : memref<f64>

    %f = constant @emlir_compute_reproj_error : (memref<11xf64>, memref<3xf64>, memref<f64>, memref<2xf64>, memref<2xf64>) -> f64
    %df = standalone.diff %f {const = [3]} : (
      memref<11xf64>,
      memref<3xf64>,
      memref<f64>,
      memref<2xf64>,
      memref<2xf64>
    ) -> f64, (
      memref<11xf64>,
      memref<11xf64>,
      memref<3xf64>,
      memref<3xf64>,
      memref<f64>,
      memref<f64>,
      memref<2xf64>,
      memref<2xf64>,
      memref<2xf64>
    ) -> f64
    call_indirect %df(%cam, %dcam, %X, %dX, %wm, %dwm, %feat, %out, %g) : (
      memref<11xf64>,
      memref<11xf64>,
      memref<3xf64>,
      memref<3xf64>,
      memref<f64>,
      memref<f64>,
      memref<2xf64>,
      memref<2xf64>,
      memref<2xf64>
    ) -> f64
    %dw = memref.load %dwm[] : memref<f64>
    return %dcam, %dX, %dw : memref<11xf64>, memref<3xf64>, f64
  }
}
