#map0 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d0)>
module  {
  memref.global "private" constant @__constant_512xf32 : memref<512xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_512x64xf32 : memref<512x64xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_10xf32 : memref<10xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_10x64xf32 : memref<10x64xf32> = dense<0.000000e+00>
  func @ebatched_cross_entropy(%arg0: memref<10x64xf32>, %arg1: memref<64xi32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 6.400000e+01 : f32
    %c64 = arith.constant 64 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.subview %arg0[0, 0] [1, 64] [1, 1] : memref<10x64xf32> to memref<64xf32, #map0>
    %1 = memref.alloca() : memref<64xf32>
    linalg.copy(%0, %1) : memref<64xf32, #map0>, memref<64xf32> 
    scf.for %arg2 = %c1 to %c10 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        %7 = memref.load %arg0[%arg2, %arg3] : memref<10x64xf32>
        %8 = memref.load %1[%arg3] : memref<64xf32>
        %9 = arith.cmpf ogt, %7, %8 : f32
        %10 = select %9, %7, %8 : f32
        memref.store %10, %1[%arg3] : memref<64xf32>
      }
    }
    %2 = memref.get_global @__constant_64xf32 : memref<64xf32>
    %3 = memref.alloc() : memref<10x64xf32>
    linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1 : memref<10x64xf32>, memref<64xf32>) outs(%3 : memref<10x64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %7 = arith.subf %arg2, %arg3 : f32
      %8 = math.exp %7 : f32
      linalg.yield %8 : f32
    }
    %4 = memref.alloca() : memref<64xf32>
    linalg.copy(%2, %4) : memref<64xf32>, memref<64xf32> 
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction", "parallel"]} ins(%3 : memref<10x64xf32>) outs(%4 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      %7 = arith.addf %arg2, %arg3 : f32
      linalg.yield %7 : f32
    }
    %5 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %cst) -> (f32) {
      %7 = memref.load %arg1[%arg2] : memref<64xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %3[%8, %arg2] : memref<10x64xf32>
      %10 = memref.load %4[%arg2] : memref<64xf32>
      %11 = arith.divf %9, %10 : f32
      %12 = math.log %11 : f32
      %13 = arith.negf %12 : f32
      %14 = arith.addf %arg3, %13 : f32
      scf.yield %14 : f32
    }
    memref.dealloc %3 : memref<10x64xf32>
    %6 = arith.divf %5, %cst_0 : f32
    return %6 : f32
  }
  // func @cross_entropy(%arg0: memref<10xf32>, %arg1: index) -> f32 {
  //   %cst = arith.constant 0.000000e+00 : f32
  //   %c10 = arith.constant 10 : index
  //   %c1 = arith.constant 1 : index
  //   %c0 = arith.constant 0 : index
  //   %0 = memref.load %arg0[%c0] : memref<10xf32>
  //   %1 = scf.for %arg2 = %c1 to %c10 step %c1 iter_args(%arg3 = %0) -> (f32) {
  //     %8 = memref.load %arg0[%arg2] : memref<10xf32>
  //     %9 = arith.cmpf ogt, %8, %arg3 : f32
  //     %10 = select %9, %8, %arg3 : f32
  //     scf.yield %10 : f32
  //   }
  //   %2 = memref.alloca() : memref<10xf32>
  //   linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg0 : memref<10xf32>) outs(%2 : memref<10xf32>) {
  //   ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
  //     %8 = arith.subf %arg2, %1 : f32
  //     %9 = math.exp %8 : f32
  //     linalg.yield %9 : f32
  //   }
  //   %3 = scf.for %arg2 = %c0 to %c10 step %c1 iter_args(%arg3 = %cst) -> (f32) {
  //     %8 = memref.load %2[%arg2] : memref<10xf32>
  //     %9 = arith.addf %arg3, %8 : f32
  //     scf.yield %9 : f32
  //   }
  //   %4 = memref.alloca() : memref<10xf32>
  //   linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%2 : memref<10xf32>) outs(%4 : memref<10xf32>) {
  //   ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
  //     %8 = arith.divf %arg2, %3 : f32
  //     linalg.yield %8 : f32
  //   }
  //   %5 = memref.load %4[%arg1] : memref<10xf32>
  //   %6 = math.log %5 : f32
  //   %7 = arith.negf %6 : f32
  //   return %7 : f32
  // }
  // func private @print_memref_f32(memref<*xf32>) attributes {llvm.emit_c_interface}
  // func private @print_memref_i32(memref<*xi32>) attributes {llvm.emit_c_interface}
  func @emlir_mlp_batched(%arg0: memref<784x64xf32>, %arg1: memref<64xi32>, %arg2: memref<512x784xf32>, %arg3: memref<512xf32>, %arg4: memref<512x512xf32>, %arg5: memref<512xf32>, %arg6: memref<10x512xf32>, %arg7: memref<10xf32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_512x64xf32 : memref<512x64xf32>
    %1 = memref.get_global @__constant_512x64xf32 : memref<512x64xf32>
    %2 = memref.get_global @__constant_10x64xf32 : memref<10x64xf32>
    %3 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%0, %3) : memref<512x64xf32>, memref<512x64xf32> 
    linalg.matmul ins(%arg2, %arg0 : memref<512x784xf32>, memref<784x64xf32>) outs(%3 : memref<512x64xf32>)
    %4 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %arg3 : memref<512x64xf32>, memref<512xf32>) outs(%4 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %12 = arith.addf %arg8, %arg9 : f32
      linalg.yield %12 : f32
    }
    memref.dealloc %3 : memref<512x64xf32>
    %5 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "ReLU 2D", indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4 : memref<512x64xf32>) outs(%5 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %12 = arith.cmpf ogt, %arg8, %cst : f32
      %13 = select %12, %arg8, %cst : f32
      linalg.yield %13 : f32
    }
    memref.dealloc %4 : memref<512x64xf32>
    %6 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%1, %6) : memref<512x64xf32>, memref<512x64xf32> 
    linalg.matmul ins(%arg4, %5 : memref<512x512xf32>, memref<512x64xf32>) outs(%6 : memref<512x64xf32>)
    memref.dealloc %5 : memref<512x64xf32>
    %7 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%6, %arg5 : memref<512x64xf32>, memref<512xf32>) outs(%7 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %12 = arith.addf %arg8, %arg9 : f32
      linalg.yield %12 : f32
    }
    memref.dealloc %6 : memref<512x64xf32>
    %8 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "ReLU 2D", indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7 : memref<512x64xf32>) outs(%8 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %12 = arith.cmpf ogt, %arg8, %cst : f32
      %13 = select %12, %arg8, %cst : f32
      linalg.yield %13 : f32
    }
    memref.dealloc %7 : memref<512x64xf32>
    %9 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%2, %9) : memref<10x64xf32>, memref<10x64xf32> 
    linalg.matmul ins(%arg6, %8 : memref<10x512xf32>, memref<512x64xf32>) outs(%9 : memref<10x64xf32>)
    memref.dealloc %8 : memref<512x64xf32>
    %10 = memref.alloc() : memref<10x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %arg7 : memref<10x64xf32>, memref<10xf32>) outs(%10 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %12 = arith.addf %arg8, %arg9 : f32
      linalg.yield %12 : f32
    }
    memref.dealloc %9 : memref<10x64xf32>
    %11 = call @ebatched_cross_entropy(%10, %arg1) : (memref<10x64xf32>, memref<64xi32>) -> f32
    memref.dealloc %10 : memref<10x64xf32>
    return %11 : f32
  }

  func @enzyme_mlir_mlp_batched(
    %arg0: memref<784x64xf32>,
    %arg1: memref<64xi32>,
    %arg2: memref<512x784xf32>,
    %arg3: memref<512xf32>,
    %arg4: memref<512x512xf32>,
    %arg5: memref<512xf32>,
    %arg6: memref<10x512xf32>,
    %arg7: memref<10xf32>
  ) -> (
    memref<512x784xf32>,
    memref<512xf32>,
    memref<512x512xf32>,
    memref<512xf32>,
    memref<10x512xf32>,
    memref<10xf32>
  ) {
    %dweight0 = memref.alloc() : memref<512x784xf32>
    %dbias0   = memref.alloc() : memref<512xf32>
    %dweight1 = memref.alloc() : memref<512x512xf32>
    %dbias1   = memref.alloc() : memref<512xf32>
    %dweight2 = memref.alloc() : memref<10x512xf32>
    %dbias2   = memref.alloc() : memref<10xf32>
    %zero = arith.constant 0.0 : f32
    linalg.fill(%zero, %dweight0) : f32, memref<512x784xf32>
    linalg.fill(%zero, %dbias0) : f32, memref<512xf32>
    linalg.fill(%zero, %dweight1) : f32, memref<512x512xf32>
    linalg.fill(%zero, %dbias1) : f32, memref<512xf32>
    linalg.fill(%zero, %dweight2) : f32, memref<10x512xf32>
    linalg.fill(%zero, %dbias2) : f32, memref<10xf32>
    %f = constant @emlir_mlp_batched : (
      memref<784x64xf32>,
      memref<64xi32>,
      memref<512x784xf32>,
      memref<512xf32>,
      memref<512x512xf32>,
      memref<512xf32>,
      memref<10x512xf32>,
      memref<10xf32>
    ) -> f32
    %df = lagrad.diff %f {const = [0]} : (
      memref<784x64xf32>,
      memref<64xi32>,
      memref<512x784xf32>,
      memref<512xf32>,
      memref<512x512xf32>,
      memref<512xf32>,
      memref<10x512xf32>,
      memref<10xf32>
    ) -> f32, (
      memref<784x64xf32>,
      memref<64xi32>,
      memref<512x784xf32>,
      memref<512x784xf32>,
      memref<512xf32>,
      memref<512xf32>,
      memref<512x512xf32>,
      memref<512x512xf32>,
      memref<512xf32>,
      memref<512xf32>,
      memref<10x512xf32>,
      memref<10x512xf32>,
      memref<10xf32>,
      memref<10xf32>
    ) -> f32
    call_indirect %df(
      %arg0, %arg1,
      %arg2, %dweight0,
      %arg3, %dbias0,
      %arg4, %dweight1,
      %arg5, %dbias1,
      %arg6, %dweight2,
      %arg7, %dbias2
    ) : (
      memref<784x64xf32>,
      memref<64xi32>,
      memref<512x784xf32>,
      memref<512x784xf32>,
      memref<512xf32>,
      memref<512xf32>,
      memref<512x512xf32>,
      memref<512x512xf32>,
      memref<512xf32>,
      memref<512xf32>,
      memref<10x512xf32>,
      memref<10x512xf32>,
      memref<10xf32>,
      memref<10xf32>
    ) -> f32

    return %dweight0, %dbias0, %dweight1, %dbias1, %dweight2, %dbias2 :
      memref<512x784xf32>,
      memref<512xf32>,
      memref<512x512xf32>,
      memref<512xf32>,
      memref<10x512xf32>,
      memref<10xf32>
  }
  // func @mlir_mlp(%arg0: memref<64x784xf32>, %arg1: memref<64xi32>, %arg2: memref<512x784xf32>, %arg3: memref<512xf32>, %arg4: memref<512x512xf32>, %arg5: memref<512xf32>, %arg6: memref<10x512xf32>, %arg7: memref<10xf32>) -> f32 {
  //   %cst = arith.constant 6.400000e+01 : f32
  //   %c64 = arith.constant 64 : index
  //   %c1 = arith.constant 1 : index
  //   %c0 = arith.constant 0 : index
  //   %cst_0 = arith.constant 0.000000e+00 : f32
  //   %0 = memref.get_global @__constant_512xf32 : memref<512xf32>
  //   %1 = memref.get_global @__constant_10xf32 : memref<10xf32>
  //   %2 = memref.alloc() : memref<512xf32>
  //   %3 = memref.alloc() : memref<512xf32>
  //   %4 = memref.alloc() : memref<512xf32>
  //   %5 = memref.alloc() : memref<512xf32>
  //   %6 = memref.alloc() : memref<512xf32>
  //   %7 = memref.alloc() : memref<512xf32>
  //   %8 = memref.alloca() : memref<10xf32>
  //   %9 = memref.alloca() : memref<10xf32>
  //   %10 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %cst_0) -> (f32) {
  //     %12 = memref.subview %arg0[%arg8, 0] [1, 784] [1, 1] : memref<64x784xf32> to memref<784xf32, #map0>
  //     %13 = memref.cast %12 : memref<784xf32, #map0> to memref<784xf32>
  //     linalg.copy(%0, %2) : memref<512xf32>, memref<512xf32> 
  //     linalg.matvec ins(%arg2, %13 : memref<512x784xf32>, memref<784xf32>) outs(%2 : memref<512xf32>)
  //     linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%2, %arg3 : memref<512xf32>, memref<512xf32>) outs(%3 : memref<512xf32>) {
  //     ^bb0(%arg10: f32, %arg11: f32, %arg12: f32):  // no predecessors
  //       %18 = arith.addf %arg10, %arg11 : f32
  //       linalg.yield %18 : f32
  //     }
  //     linalg.generic {doc = "ReLU", indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%3 : memref<512xf32>) outs(%4 : memref<512xf32>) {
  //     ^bb0(%arg10: f32, %arg11: f32):  // no predecessors
  //       %18 = arith.cmpf ogt, %arg10, %cst_0 : f32
  //       %19 = select %18, %arg10, %cst_0 : f32
  //       linalg.yield %19 : f32
  //     }
  //     linalg.copy(%0, %5) : memref<512xf32>, memref<512xf32> 
  //     linalg.matvec ins(%arg4, %4 : memref<512x512xf32>, memref<512xf32>) outs(%5 : memref<512xf32>)
  //     linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%5, %arg5 : memref<512xf32>, memref<512xf32>) outs(%6 : memref<512xf32>) {
  //     ^bb0(%arg10: f32, %arg11: f32, %arg12: f32):  // no predecessors
  //       %18 = arith.addf %arg10, %arg11 : f32
  //       linalg.yield %18 : f32
  //     }
  //     linalg.generic {doc = "ReLU", indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%6 : memref<512xf32>) outs(%7 : memref<512xf32>) {
  //     ^bb0(%arg10: f32, %arg11: f32):  // no predecessors
  //       %18 = arith.cmpf ogt, %arg10, %cst_0 : f32
  //       %19 = select %18, %arg10, %cst_0 : f32
  //       linalg.yield %19 : f32
  //     }
  //     linalg.copy(%1, %8) : memref<10xf32>, memref<10xf32> 
  //     linalg.matvec ins(%arg6, %7 : memref<10x512xf32>, memref<512xf32>) outs(%8 : memref<10xf32>)
  //     linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%8, %arg7 : memref<10xf32>, memref<10xf32>) outs(%9 : memref<10xf32>) {
  //     ^bb0(%arg10: f32, %arg11: f32, %arg12: f32):  // no predecessors
  //       %18 = arith.addf %arg10, %arg11 : f32
  //       linalg.yield %18 : f32
  //     }
  //     %14 = memref.load %arg1[%arg8] : memref<64xi32>
  //     %15 = arith.index_cast %14 : i32 to index
  //     %16 = call @cross_entropy(%9, %15) : (memref<10xf32>, index) -> f32
  //     %17 = arith.addf %arg9, %16 : f32
  //     scf.yield %17 : f32
  //   }
  //   memref.dealloc %7 : memref<512xf32>
  //   memref.dealloc %6 : memref<512xf32>
  //   memref.dealloc %5 : memref<512xf32>
  //   memref.dealloc %4 : memref<512xf32>
  //   memref.dealloc %3 : memref<512xf32>
  //   memref.dealloc %2 : memref<512xf32>
  //   %11 = arith.divf %10, %cst : f32
  //   return %11 : f32
  // }
}

