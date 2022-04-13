// #map0 = affine_map<(d0, d1) -> (d0, d1)>
// #map1 = affine_map<(d0, d1) -> (d0)>
// #map2 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
// #map3 = affine_map<(d0) -> (d0)>
// #map4 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s0 + d1 * s2 + s1)>
// #map5 = affine_map<(d0) -> ()>

// memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
// memref.global "private" constant @__constant_2xf64 : memref<2xf64> = dense<0.000000e+00>
// memref.global "private" constant @__constant_4xf64 : memref<4xf64> = dense<0.000000e+00>

// func private @print_memref_f64(memref<*xf64>) attributes {llvm.emit_c_interface}
// func @emain_term(%arg0: memref<4xf64>, %arg1: memref<4x2xf64>, %arg2: memref<4x2xf64>, %arg3: memref<4x2x2xf64>, %arg4: memref<10x2xf64>, %out: memref<f64>) -> f64 {
//   %cst = arith.constant 0.000000e+00 : f64
//   %0 = memref.alloc() : memref<4x2xf64>
//   %1 = memref.get_global @__constant_4xf64 : memref<4xf64>
//   %2 = memref.get_global @__constant_2xf64 : memref<2xf64>
//   %3 = memref.alloc() : memref<4xf64>
//   %4 = memref.get_global @__constant_xf64 : memref<f64>
//   %5 = memref.alloc() : memref<f64>
//   %6 = memref.alloc() : memref<4x2xf64>
//   linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<4x2xf64>) outs(%6 : memref<4x2xf64>) {
//   ^bb0(%arg5: f64, %arg6: f64):  // no predecessors
//     %9 = math.exp %arg5 : f64
//     linalg.yield %9 : f64
//   }
//   %7 = memref.alloc() : memref<4xf64>
//   linalg.copy(%1, %7) : memref<4xf64>, memref<4xf64>
//   linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<4x2xf64>) outs(%7 : memref<4xf64>) {
//   ^bb0(%arg5: f64, %arg6: f64):  // no predecessors
//     %9 = arith.addf %arg5, %arg6 : f64
//     linalg.yield %9 : f64
//   }
//   %cst_0 = arith.constant 5.000000e-01 : f64
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//   %c10 = arith.constant 10 : index
//   %c4 = arith.constant 4 : index
//   %c2_1 = arith.constant 2 : index
//   %8 = scf.for %arg5 = %c0 to %c10 step %c1 iter_args(%arg6 = %cst) -> (f64) {
//     %9 = scf.for %arg7 = %c0 to %c4 step %c1 iter_args(%arg8 = %3) -> (memref<4xf64>) {
//       %18 = memref.subview %arg4[%arg5, 0] [1, 2] [1, 1] : memref<10x2xf64> to memref<2xf64, #map2>
//       %19 = memref.cast %18 : memref<2xf64, #map2> to memref<2xf64>
//       %20 = memref.subview %arg1[%arg7, 0] [1, 2] [1, 1] : memref<4x2xf64> to memref<2xf64, #map2>
//       %21 = memref.cast %20 : memref<2xf64, #map2> to memref<2xf64>
//       %22 = memref.alloc() : memref<2xf64>
//       linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%19, %21 : memref<2xf64>, memref<2xf64>) outs(%22 : memref<2xf64>) {
//       ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
//         %37 = arith.subf %arg9, %arg10 : f64
//         linalg.yield %37 : f64
//       }
//       %23 = memref.subview %6[%arg7, 0] [1, 2] [1, 1] : memref<4x2xf64> to memref<2xf64, #map2>
//       %24 = memref.cast %23 : memref<2xf64, #map2> to memref<2xf64>
//       %25 = memref.subview %arg3[%arg7, 0, 0] [1, 2, 2] [1, 1, 1] : memref<4x2x2xf64> to memref<2x2xf64, #map4>
//       %26 = memref.cast %25 : memref<2x2xf64, #map4> to memref<2x2xf64>
//       %27 = memref.alloc() : memref<2xf64>
//       linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%24, %22 : memref<2xf64>, memref<2xf64>) outs(%27 : memref<2xf64>) {
//       ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
//         %37 = arith.mulf %arg9, %arg10 : f64
//         linalg.yield %37 : f64
//       }
//       %28 = memref.alloc() : memref<2xf64>
//       linalg.copy(%2, %28) : memref<2xf64>, memref<2xf64>
//       linalg.matvec ins(%26, %22 : memref<2x2xf64>, memref<2xf64>) outs(%28 : memref<2xf64>)
//       %29 = memref.alloc() : memref<2xf64>
//       linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%27, %28 : memref<2xf64>, memref<2xf64>) outs(%29 : memref<2xf64>) {
//       ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
//         %37 = arith.addf %arg9, %arg10 : f64
//         linalg.yield %37 : f64
//       }
//       %30 = memref.alloc() : memref<f64>
//       linalg.copy(%4, %30) : memref<f64>, memref<f64>
//       linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["reduction"]} ins(%29 : memref<2xf64>) outs(%30 : memref<f64>) {
//       ^bb0(%arg9: f64, %arg10: f64):  // no predecessors
//         %37 = arith.mulf %arg9, %arg9 : f64
//         %38 = arith.addf %37, %arg10 : f64
//         linalg.yield %38 : f64
//       }
//       %31 = memref.load %30[] : memref<f64>
//       %32 = arith.mulf %31, %cst_0 : f64
//       %33 = memref.load %arg0[%arg7] : memref<4xf64>
//       %34 = memref.load %7[%arg7] : memref<4xf64>
//       %35 = arith.addf %33, %34 : f64
//       %36 = arith.subf %35, %32 : f64
//       memref.store %36, %arg8[%arg7] : memref<4xf64>
//       scf.yield %arg8 : memref<4xf64>
//     }
//     %10 = memref.load %9[%c0] : memref<4xf64>
//     memref.store %10, %5[] : memref<f64>
//     %11 = memref.alloc() : memref<f64>
//     linalg.copy(%5, %11) : memref<f64>, memref<f64>
//     linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["reduction"]} ins(%9 : memref<4xf64>) outs(%11 : memref<f64>) {
//     ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
//       %18 = arith.cmpf ogt, %arg7, %arg8 : f64
//       %19 = scf.if %18 -> (f64) {
//         scf.yield %arg7 : f64
//       } else {
//         scf.yield %arg8 : f64
//       }
//       linalg.yield %19 : f64
//     }
//     %12 = memref.load %11[] : memref<f64>
//     %13 = memref.alloc() : memref<f64>
//     linalg.copy(%4, %13) : memref<f64>, memref<f64>
//     linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["reduction"]} ins(%9 : memref<4xf64>) outs(%13 : memref<f64>) {
//     ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
//       %18 = arith.subf %arg7, %12 : f64
//       %19 = math.exp %18 : f64
//       %20 = arith.addf %19, %arg8 : f64
//       linalg.yield %20 : f64
//     }
//     %14 = memref.load %13[] : memref<f64>
//     %15 = math.log %14 : f64
//     %16 = arith.addf %15, %12 : f64
//     %17 = arith.addf %arg6, %16 : f64
//     scf.yield %17 : f64
//   }
//   memref.store %8, %out[] : memref<f64>
//   // %ret = arith.constant 0.0 : f64
//   return %8 : f64
// }

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s0 + d1 * s2 + s1)>
#map5 = affine_map<(d0) -> ()>
// module  {
memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
memref.global "private" constant @__constant_dxf64 : memref<{{d}}xf64> = dense<0.000000e+00>
memref.global "private" constant @__constant_kxf64 : memref<{{k}}xf64> = dense<0.000000e+00>
func @emain_term(%arg0: memref<{{k}}xf64>, %arg1: memref<{{k}}x{{d}}xf64>, %arg2: memref<{{k}}x{{d}}xf64>, %arg3: memref<{{k}}x{{d}}x{{d}}xf64>, %arg4: memref<{{n}}x{{d}}xf64>) -> f64 {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %1 = memref.get_global @__constant_kxf64 : memref<{{k}}xf64>
  %2 = memref.get_global @__constant_dxf64 : memref<{{d}}xf64>
  %3 = memref.alloc() : memref<{{k}}xf64>
  %4 = memref.get_global @__constant_xf64 : memref<f64>
  %5 = memref.alloc() : memref<f64>
  %6 = memref.alloc() : memref<{{k}}x{{d}}xf64>
  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%6 : memref<{{k}}x{{d}}xf64>) {
  ^bb0(%arg5: f64, %arg6: f64):  // no predecessors
    %9 = math.exp %arg5 : f64
    linalg.yield %9 : f64
  }
  %7 = memref.alloc() : memref<{{k}}xf64>
  linalg.copy(%1, %7) : memref<{{k}}xf64>, memref<{{k}}xf64>
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<{{k}}x{{d}}xf64>) outs(%7 : memref<{{k}}xf64>) {
  ^bb0(%arg5: f64, %arg6: f64):  // no predecessors
    %9 = arith.addf %arg5, %arg6 : f64
    linalg.yield %9 : f64
  }
  %cst_0 = arith.constant 5.000000e-01 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c256 = arith.constant {{n}} : index
  %c128 = arith.constant {{k}} : index
  %c64 = arith.constant {{d}} : index
  %8 = scf.for %arg5 = %c0 to %c256 step %c1 iter_args(%arg6 = %cst) -> (f64) {
    %9 = scf.for %arg7 = %c0 to %c128 step %c1 iter_args(%arg8 = %3) -> (memref<{{k}}xf64>) {
      %19 = memref.subview %arg4[%arg5, 0] [1, {{d}}] [1, 1] : memref<{{n}}x{{d}}xf64> to memref<{{d}}xf64, #map2>
      // %19 = memref.cast %18 : memref<{{d}}xf64, #map2> to memref<{{d}}xf64>
      %21 = memref.subview %arg1[%arg7, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map2>
      // %21 = memref.cast %20 : memref<{{d}}xf64, #map2> to memref<{{d}}xf64>
      %22 = memref.alloc() : memref<{{d}}xf64>
      linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%19, %21 : memref<{{d}}xf64, #map2>, memref<{{d}}xf64, #map2>) outs(%22 : memref<{{d}}xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
        %37 = arith.subf %arg9, %arg10 : f64
        linalg.yield %37 : f64
      }
      // %xcmemref = memref.alloc() : memref<f64>
      // memref.store %cst, %xcmemref[] : memref<f64>
      // linalg.generic
      //   {indexing_maps = [#map3, #map5], iterator_types = ["reduction"]}
      //   ins(%22 : memref<{{d}}xf64>)
      //   outs(%xcmemref : memref<f64>) {
      // ^bb0(%g0: f64, %g1: f64):
      //   %interm0 = arith.addf %g0, %g1 : f64
      //   linalg.yield %interm0 : f64
      // }
      // %mt0 = memref.load %xcmemref[] : memref<f64>
      %aik = memref.load %arg0[%arg7] : memref<{{k}}xf64>
      // %36 = arith.subf %mt0, %aik : f64
      %24 = memref.subview %6[%arg7, 0] [1, {{d}}] [1, 1] : memref<{{k}}x{{d}}xf64> to memref<{{d}}xf64, #map2>
      // %24 = memref.cast %23 : memref<{{d}}xf64, #map2> to memref<{{d}}xf64>
      %26 = memref.subview %arg3[%arg7, 0, 0] [1, 64, {{d}}] [1, 1, 1] : memref<{{k}}x{{d}}x{{d}}xf64> to memref<{{d}}x{{d}}xf64, #map4>
      // %26 = memref.cast %25 : memref<{{d}}x{{d}}xf64, #map4> to memref<{{d}}x{{d}}xf64>
      %27 = memref.alloc() : memref<{{d}}xf64>
      linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%24, %22 : memref<{{d}}xf64, #map2>, memref<{{d}}xf64>) outs(%27 : memref<{{d}}xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
        %37 = arith.mulf %arg9, %arg10 : f64
        linalg.yield %37 : f64
      }

      %28 = memref.alloc() : memref<{{d}}xf64>
      linalg.fill(%cst, %28) : f64, memref<{{d}}xf64>
      linalg.matvec ins(%26, %22 : memref<{{d}}x{{d}}xf64, #map4>, memref<{{d}}xf64>) outs(%28 : memref<{{d}}xf64>)
      %29 = memref.alloc() : memref<{{d}}xf64>
      linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%27, %28 : memref<{{d}}xf64>, memref<{{d}}xf64>) outs(%29 : memref<{{d}}xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
        %37 = arith.addf %arg9, %arg10 : f64
        linalg.yield %37 : f64
      }

      // %reduced = memref.alloc() : memref<f64>
      // memref.store %cst, %reduced[] : memref<f64>
      // linalg.generic
      //   {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]}
      //   ins(%28 : memref<{{d}}xf64>)
      //   outs(%reduced : memref<f64>) {
      // ^bb0(%g0: f64, %g1: f64):
      //   %i0 = arith.addf %g0, %g1 : f64
      //   linalg.yield %i0 : f64
      // }
      // %rval = memref.load %reduced[] : memref<f64>
      // %36 = arith.addf %rval, %aik : f64
      %30 = memref.alloc() : memref<f64>
      linalg.copy(%4, %30) : memref<f64>, memref<f64>
      linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["reduction"]} ins(%29 : memref<{{d}}xf64>) outs(%30 : memref<f64>) {
      ^bb0(%arg9: f64, %arg10: f64):  // no predecessors
        %37 = arith.mulf %arg9, %arg9 : f64
        %38 = arith.addf %37, %arg10 : f64
        linalg.yield %38 : f64
      }
      %31 = memref.load %30[] : memref<f64>
      %32 = arith.mulf %31, %cst_0 : f64
      %33 = memref.load %arg0[%arg7] : memref<{{k}}xf64>
      %34 = memref.load %7[%arg7] : memref<{{k}}xf64>
      %35 = arith.addf %33, %34 : f64
      %36 = arith.subf %35, %32 : f64
      memref.store %36, %arg8[%arg7] : memref<{{k}}xf64>
      scf.yield %arg8 : memref<{{k}}xf64>
    }
    // memref.store %cst, %5[] : memref<f64>
    // linalg.generic
    //   {
    //     indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
    //     iterator_types = ["reduction"]
    //   }
    //   ins(%9 : memref<{{k}}xf64>)
    //   outs(%5 : memref<f64>) {
    // ^bb0(%arg9: f64, %arg10: f64):
    //   %19 = arith.addf %arg9, %arg10 : f64
    //   linalg.yield %19 : f64
    // }
    // %16 = memref.load %5[] : memref<f64>
    %10 = memref.load %9[%c0] : memref<{{k}}xf64>
    memref.store %10, %5[] : memref<f64>
    %11 = memref.alloc() : memref<f64>
    linalg.copy(%5, %11) : memref<f64>, memref<f64>
    linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["reduction"]} ins(%9 : memref<{{k}}xf64>) outs(%11 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %18 = arith.cmpf ogt, %arg7, %arg8 : f64
      %19 = scf.if %18 -> (f64) {
        scf.yield %arg7 : f64
      } else {
        scf.yield %arg8 : f64
      }
      linalg.yield %19 : f64
    }
    %12 = memref.load %11[] : memref<f64>
    %13 = memref.alloc() : memref<f64>
    linalg.copy(%4, %13) : memref<f64>, memref<f64>
    linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["reduction"]} ins(%9 : memref<{{k}}xf64>) outs(%13 : memref<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %18 = arith.subf %arg7, %12 : f64
      %19 = math.exp %18 : f64
      %20 = arith.addf %19, %arg8 : f64
      linalg.yield %20 : f64
    }
    %14 = memref.load %13[] : memref<f64>
    %15 = math.log %14 : f64
    %16 = arith.addf %15, %12 : f64
    %17 = arith.addf %arg6, %16 : f64
    scf.yield %17 : f64
  }
  return %8 : f64
}

func @main_term_enzyme(
  %arg0: memref<{{k}}xf64>,
  %arg1: memref<{{k}}x{{d}}xf64>,
  %arg2: memref<{{k}}x{{d}}xf64>,
  %arg3: memref<{{k}}x{{d}}x{{d}}xf64>,
  %arg4: memref<{{n}}x{{d}}xf64>
) -> (
  memref<{{k}}xf64>,
  memref<{{k}}x{{d}}xf64>,
  memref<{{k}}x{{d}}xf64>,
  memref<{{k}}x{{d}}x{{d}}xf64>
) {
  %darg0 = memref.alloc() : memref<{{k}}xf64>
  %darg1 = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %darg2 = memref.alloc() : memref<{{k}}x{{d}}xf64>
  %darg3 = memref.alloc() : memref<{{k}}x{{d}}x{{d}}xf64>
  %zero = arith.constant 0.0 : f64

  linalg.fill(%zero, %darg0) : f64, memref<{{k}}xf64>
  linalg.fill(%zero, %darg1) : f64, memref<{{k}}x{{d}}xf64>
  linalg.fill(%zero, %darg2) : f64, memref<{{k}}x{{d}}xf64>
  linalg.fill(%zero, %darg3) : f64, memref<{{k}}x{{d}}x{{d}}xf64>

  %f = constant @emain_term : (
    memref<{{k}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}x{{d}}xf64>,
    memref<{{n}}x{{d}}xf64>
  ) -> f64
  %df = standalone.diff %f {const = [4]} : (
    memref<{{k}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}x{{d}}xf64>,
    memref<{{n}}x{{d}}xf64>
  ) -> f64, (
    memref<{{k}}xf64>,
    memref<{{k}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}x{{d}}xf64>,
    memref<{{k}}x{{d}}x{{d}}xf64>,
    memref<{{n}}x{{d}}xf64>
  ) -> f64
  call_indirect %df(%arg0, %darg0, %arg1, %darg1, %arg2, %darg2, %arg3, %darg3, %arg4) : (
    memref<{{k}}xf64>,
    memref<{{k}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}x{{d}}xf64>,
    memref<{{k}}x{{d}}x{{d}}xf64>,
    memref<{{n}}x{{d}}xf64>
  ) -> f64

  return %darg0, %darg1, %darg2, %darg3 : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}x{{d}}xf64>
}
// }
