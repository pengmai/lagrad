#map0 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0) -> ()>
#map6 = affine_map<(d0, d1) -> (d0)>
module  {
  func private @smatvec(memref<?x?xf32, #map0>, memref<?xf32, #map1>, memref<?xf32, #map1>) attributes {llvm.emit_c_interface}
  func private @svecmat(memref<?xf32, #map1>, memref<?x?xf32, #map0>, memref<?xf32, #map1>) attributes {llvm.emit_c_interface}
  func private @souter(memref<?xf32, #map1>, memref<?xf32, #map1>, memref<?x?xf32, #map0>) attributes {llvm.emit_c_interface}
  func private @smatmul(memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) attributes {llvm.emit_c_interface}
  func private @smatmul_grad_second(memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) attributes {llvm.emit_c_interface}
  func private @smatmul_grad_first(memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) attributes {llvm.emit_c_interface}
  func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
  memref.global "private" constant @__constant_512xf32 : memref<512xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_512x64xf32 : memref<512x64xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf32 : memref<f32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_10xf32 : memref<10xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_10x64xf32 : memref<10x64xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<0.000000e+00>
  func @batched_cross_entropy(%arg0: memref<10x64xf32>, %arg1: memref<64xi32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 6.400000e+01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_64xf32 : memref<64xf32>
    %1 = memref.subview %arg0[0, 0] [1, 64] [1, 1] : memref<10x64xf32> to memref<64xf32, #map1>
    %2 = memref.alloca() : memref<64xf32>
    linalg.copy(%1, %2) : memref<64xf32, #map1>, memref<64xf32> 
    scf.for %arg2 = %c1 to %c10 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        %7 = memref.load %arg0[%arg2, %arg3] : memref<10x64xf32>
        %8 = memref.load %2[%arg3] : memref<64xf32>
        %9 = arith.cmpf ogt, %7, %8 : f32
        %10 = select %9, %7, %8 : f32
        memref.store %10, %2[%arg3] : memref<64xf32>
      }
    }
    %3 = memref.alloc() : memref<10x64xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : memref<10x64xf32>, memref<64xf32>) outs(%3 : memref<10x64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %7 = arith.subf %arg2, %arg3 : f32
      %8 = math.exp %7 : f32
      linalg.yield %8 : f32
    }
    %4 = memref.alloca() : memref<64xf32>
    linalg.copy(%0, %4) : memref<64xf32>, memref<64xf32> 
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel"]} ins(%3 : memref<10x64xf32>) outs(%4 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      %7 = arith.addf %arg2, %arg3 : f32
      linalg.yield %7 : f32
    }
    %5 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %cst_0) -> (f32) {
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
    %6 = arith.divf %5, %cst : f32
    return %6 : f32
  }
  func @__grad_batched_cross_entropy_arg0(%arg0: memref<10x64xf32>, %arg1: memref<64xi32>, %arg2: f32) -> memref<10x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 6.400000e+01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c63 = arith.constant 63 : index
    %0 = memref.get_global @__constant_64xf32 : memref<64xf32>
    %1 = memref.get_global @__constant_10x64xf32 : memref<10x64xf32>
    %2 = memref.subview %arg0[0, 0] [1, 64] [1, 1] : memref<10x64xf32> to memref<64xf32, #map1>
    %3 = memref.alloca() : memref<64xf32>
    linalg.copy(%2, %3) : memref<64xf32, #map1>, memref<64xf32> 
    %4 = memref.alloca() : memref<64x9xi1>
    scf.for %arg3 = %c1 to %c10 step %c1 {
      scf.for %arg4 = %c0 to %c64 step %c1 {
        %16 = memref.load %arg0[%arg3, %arg4] : memref<10x64xf32>
        %17 = memref.load %3[%arg4] : memref<64xf32>
        %18 = arith.cmpf ogt, %16, %17 : f32
        memref.store %18, %4[%arg4, %arg3] {lagrad_cache} : memref<64x9xi1>
        %19 = select %18, %16, %17 : f32
        memref.store %19, %3[%arg4] : memref<64xf32>
      }
    }
    %5 = memref.alloc() : memref<10x64xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0, %3 : memref<10x64xf32>, memref<64xf32>) outs(%5 : memref<10x64xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %16 = arith.subf %arg3, %arg4 : f32
      %17 = math.exp %16 : f32
      linalg.yield %17 : f32
    }
    %6 = memref.alloca() : memref<64xf32>
    linalg.copy(%0, %6) : memref<64xf32>, memref<64xf32> 
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel"]} ins(%5 : memref<10x64xf32>) outs(%6 : memref<64xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      %16 = arith.addf %arg3, %arg4 : f32
      linalg.yield %16 : f32
    }
    %7 = arith.divf %arg2, %cst : f32
    %8 = memref.alloca() : memref<64xf32>
    linalg.fill(%cst_0, %8) : f32, memref<64xf32> 
    %9 = memref.alloc() : memref<10x64xf32>
    linalg.fill(%cst_0, %9) : f32, memref<10x64xf32> 
    scf.for %arg3 = %c0 to %c64 step %c1 {
      %16 = arith.subi %c63, %arg3 : index
      %17 = memref.load %arg1[%16] : memref<64xi32>
      %18 = arith.index_cast %17 {"cloned "} : i32 to index
      %19 = memref.load %5[%18, %16] : memref<10x64xf32>
      %20 = memref.load %6[%16] : memref<64xf32>
      %21 = arith.divf %19, %20 {"cloned "} : f32
      %22 = arith.negf %7 : f32
      %23 = arith.divf %22, %21 : f32
      %24 = arith.divf %23, %20 : f32
      %25 = arith.mulf %23, %19 : f32
      %26 = arith.negf %25 : f32
      %27 = arith.mulf %20, %20 : f32
      %28 = arith.divf %26, %27 : f32
      %29 = memref.load %8[%16] : memref<64xf32>
      %30 = arith.addf %29, %28 : f32
      memref.store %30, %8[%16] : memref<64xf32>
      %31 = memref.load %9[%18, %16] : memref<10x64xf32>
      %32 = arith.addf %31, %24 : f32
      memref.store %32, %9[%18, %16] : memref<10x64xf32>
    }
    memref.dealloc %5 : memref<10x64xf32>
    %10 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%1, %10) : memref<10x64xf32>, memref<10x64xf32> 
    linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%8 : memref<64xf32>) outs(%10 : memref<10x64xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      %16 = arith.addf %arg3, %arg4 : f32
      linalg.yield %16 : f32
    }
    %11 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%9, %11) : memref<10x64xf32>, memref<10x64xf32> 
    memref.dealloc %9 : memref<10x64xf32>
    linalg.generic {doc = "Add in place", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%10 : memref<10x64xf32>) outs(%11 : memref<10x64xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      %16 = arith.addf %arg3, %arg4 : f32
      linalg.yield %16 : f32
    }
    memref.dealloc %10 : memref<10x64xf32>
    %12 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%1, %12) : memref<10x64xf32>, memref<10x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map3, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0, %3, %11 : memref<10x64xf32>, memref<64xf32>, memref<10x64xf32>) outs(%12 : memref<10x64xf32>) attrs =  {"adjoint of ", "gradient space for "} {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %16 = arith.subf %arg3, %arg4 : f32
      %17 = math.exp %16 : f32
      %18 = arith.mulf %arg5, %17 : f32
      %19 = arith.addf %18, %arg6 : f32
      linalg.yield %19 : f32
    }
    %13 = memref.alloca() : memref<64xf32>
    linalg.copy(%0, %13) : memref<64xf32>, memref<64xf32> 
    linalg.generic {indexing_maps = [#map2, #map3, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0, %3, %11 : memref<10x64xf32>, memref<64xf32>, memref<10x64xf32>) outs(%13 : memref<64xf32>) attrs =  {"adjoint of "} {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %16 = arith.subf %arg3, %arg4 : f32
      %17 = math.exp %16 : f32
      %18 = arith.mulf %arg5, %17 : f32
      %19 = arith.negf %18 : f32
      %20 = arith.addf %19, %arg6 : f32
      linalg.yield %20 : f32
    }
    memref.dealloc %11 : memref<10x64xf32>
    scf.for %arg3 = %c1 to %c10 step %c1 {
      %16 = arith.subi %c10, %arg3 : index
      scf.for %arg4 = %c0 to %c64 step %c1 {
        %17 = arith.subi %c63, %arg4 : index
        %18 = memref.load %4[%17, %16] {"cached "} : memref<64x9xi1>
        %19 = memref.load %13[%17] : memref<64xf32>
        memref.store %cst_0, %13[%17] : memref<64xf32>
        %20 = select %18, %19, %cst_0 : f32
        %21 = select %18, %cst_0, %19 : f32
        %22 = memref.load %13[%17] : memref<64xf32>
        %23 = arith.addf %22, %21 : f32
        memref.store %23, %13[%17] : memref<64xf32>
        %24 = memref.load %12[%16, %17] : memref<10x64xf32>
        %25 = arith.addf %24, %20 : f32
        memref.store %25, %12[%16, %17] : memref<10x64xf32>
      }
    }
    %14 = memref.subview %12[0, 0] [1, 64] [1, 1] : memref<10x64xf32> to memref<64xf32, #map1>
    %15 = memref.cast %14 : memref<64xf32, #map1> to memref<64xf32>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%15, %13 : memref<64xf32>, memref<64xf32>) outs(%14 : memref<64xf32, #map1>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %16 = arith.addf %arg3, %arg4 : f32
      linalg.yield %16 : f32
    }
    return %12 : memref<10x64xf32>
  }
  func @cross_entropy(%arg0: memref<10xf32>, %arg1: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.load %arg0[%c0] : memref<10xf32>
    %1 = scf.for %arg2 = %c1 to %c10 step %c1 iter_args(%arg3 = %0) -> (f32) {
      %8 = memref.load %arg0[%arg2] : memref<10xf32>
      %9 = arith.cmpf ogt, %8, %arg3 : f32
      %10 = select %9, %8, %arg3 : f32
      scf.yield %10 : f32
    }
    %2 = memref.alloca() : memref<10xf32>
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%arg0 : memref<10xf32>) outs(%2 : memref<10xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      %8 = arith.subf %arg2, %1 : f32
      %9 = math.exp %8 : f32
      linalg.yield %9 : f32
    }
    %3 = scf.for %arg2 = %c0 to %c10 step %c1 iter_args(%arg3 = %cst) -> (f32) {
      %8 = memref.load %2[%arg2] : memref<10xf32>
      %9 = arith.addf %arg3, %8 : f32
      scf.yield %9 : f32
    }
    %4 = memref.alloca() : memref<10xf32>
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%2 : memref<10xf32>) outs(%4 : memref<10xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      %8 = arith.divf %arg2, %3 : f32
      linalg.yield %8 : f32
    }
    %5 = memref.load %4[%arg1] : memref<10xf32>
    %6 = math.log %5 : f32
    %7 = arith.negf %6 : f32
    return %7 : f32
  }
  func @__grad_cross_entropy_arg0(%arg0: memref<10xf32>, %arg1: index, %arg2: f32) -> memref<10xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c9 = arith.constant 9 : index
    %0 = memref.get_global @__constant_10xf32 : memref<10xf32>
    %1 = memref.get_global @__constant_xf32 : memref<f32>
    %2 = memref.load %arg0[%c0] : memref<10xf32>
    %3 = memref.alloca() : memref<9xi1>
    %4 = scf.for %arg3 = %c1 to %c10 step %c1 iter_args(%arg4 = %2) -> (f32) {
      %21 = memref.load %arg0[%arg3] : memref<10xf32>
      %22 = arith.cmpf ogt, %21, %arg4 : f32
      memref.store %22, %3[%arg3] {lagrad_cache} : memref<9xi1>
      %23 = select %22, %21, %arg4 : f32
      scf.yield %23 : f32
    }
    %5 = memref.alloca() : memref<10xf32>
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%arg0 : memref<10xf32>) outs(%5 : memref<10xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      %21 = arith.subf %arg3, %4 : f32
      %22 = math.exp %21 : f32
      linalg.yield %22 : f32
    }
    %6 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %cst) -> (f32) {
      %21 = memref.load %5[%arg3] : memref<10xf32>
      %22 = arith.addf %arg4, %21 : f32
      scf.yield %22 : f32
    }
    %7 = memref.alloca() : memref<10xf32>
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%5 : memref<10xf32>) outs(%7 : memref<10xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      %21 = arith.divf %arg3, %6 : f32
      linalg.yield %21 : f32
    }
    %8 = memref.load %7[%arg1] : memref<10xf32>
    %9 = arith.negf %arg2 : f32
    %10 = arith.divf %9, %8 : f32
    %11 = memref.alloca() : memref<10xf32>
    linalg.fill(%cst, %11) : f32, memref<10xf32> 
    memref.store %10, %11[%arg1] : memref<10xf32>
    %12 = memref.alloca() : memref<f32>
    linalg.copy(%1, %12) : memref<f32>, memref<f32> 
    linalg.generic {indexing_maps = [#map4, #map4, #map5], iterator_types = ["parallel"]} ins(%5, %11 : memref<10xf32>, memref<10xf32>) outs(%12 : memref<f32>) attrs =  {"adjoint of "} {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %21 = arith.mulf %arg4, %arg3 : f32
      %22 = arith.negf %21 : f32
      %23 = arith.mulf %6, %6 : f32
      %24 = arith.divf %22, %23 : f32
      %25 = arith.addf %24, %arg5 : f32
      linalg.yield %25 : f32
    }
    %13 = memref.load %12[] : memref<f32>
    %14 = memref.alloca() : memref<10xf32>
    linalg.copy(%0, %14) : memref<10xf32>, memref<10xf32> 
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%11 : memref<10xf32>) outs(%14 : memref<10xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      %21 = arith.divf %arg3, %6 : f32
      %22 = arith.addf %21, %arg4 : f32
      linalg.yield %22 : f32
    }
    scf.for %arg3 = %c0 to %c10 step %c1 {
      %21 = arith.subi %c9, %arg3 : index
      %22 = memref.load %14[%21] : memref<10xf32>
      %23 = arith.addf %22, %13 : f32
      memref.store %23, %14[%21] : memref<10xf32>
    }
    %15 = memref.alloca() : memref<f32>
    linalg.copy(%1, %15) : memref<f32>, memref<f32> 
    linalg.generic {indexing_maps = [#map4, #map4, #map5], iterator_types = ["parallel"]} ins(%arg0, %14 : memref<10xf32>, memref<10xf32>) outs(%15 : memref<f32>) attrs =  {"adjoint of "} {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %21 = arith.subf %arg3, %4 : f32
      %22 = math.exp %21 : f32
      %23 = arith.mulf %arg4, %22 : f32
      %24 = arith.negf %23 : f32
      %25 = arith.addf %24, %arg5 : f32
      linalg.yield %25 : f32
    }
    %16 = memref.load %15[] : memref<f32>
    %17 = memref.alloc() : memref<10xf32>
    linalg.copy(%0, %17) : memref<10xf32>, memref<10xf32> 
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%arg0, %14 : memref<10xf32>, memref<10xf32>) outs(%17 : memref<10xf32>) attrs =  {"adjoint of ", "gradient space for "} {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %21 = arith.subf %arg3, %4 : f32
      %22 = math.exp %21 : f32
      %23 = arith.mulf %arg4, %22 : f32
      %24 = arith.addf %23, %arg5 : f32
      linalg.yield %24 : f32
    }
    %18 = scf.for %arg3 = %c1 to %c10 step %c1 iter_args(%arg4 = %16) -> (f32) {
      %21 = arith.subi %c10, %arg3 : index
      %22 = memref.load %3[%21] {"cached "} : memref<9xi1>
      %23 = select %22, %arg4, %cst : f32
      %24 = select %22, %cst, %arg4 : f32
      %25 = memref.load %17[%21] : memref<10xf32>
      %26 = arith.addf %25, %23 : f32
      memref.store %26, %17[%21] : memref<10xf32>
      scf.yield %24 : f32
    }
    %19 = memref.load %17[%c0] : memref<10xf32>
    %20 = arith.addf %19, %18 : f32
    memref.store %20, %17[%c0] : memref<10xf32>
    return %17 : memref<10xf32>
  }
  func @mlir_mlp_batched(%arg0: memref<784x64xf32>, %arg1: memref<64xi32>, %arg2: memref<512x784xf32>, %arg3: memref<512xf32>, %arg4: memref<512x512xf32>, %arg5: memref<512xf32>, %arg6: memref<10x512xf32>, %arg7: memref<10xf32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 6.400000e+01 : f32
    %c64 = arith.constant 64 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_512x64xf32 : memref<512x64xf32>
    %1 = memref.get_global @__constant_10x64xf32 : memref<10x64xf32>
    %2 = memref.get_global @__constant_64xf32 : memref<64xf32>
    %3 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%0, %3) : memref<512x64xf32>, memref<512x64xf32> 
    %4 = memref.cast %arg2 : memref<512x784xf32> to memref<?x?xf32, #map0>
    %5 = memref.cast %arg0 : memref<784x64xf32> to memref<?x?xf32, #map0>
    %6 = memref.cast %3 : memref<512x64xf32> to memref<?x?xf32, #map0>
    call @smatmul(%4, %5, %6) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    %7 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel"]} ins(%3, %arg3 : memref<512x64xf32>, memref<512xf32>) outs(%7 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %26 = arith.addf %arg8, %arg9 : f32
      linalg.yield %26 : f32
    }
    memref.dealloc %3 : memref<512x64xf32>
    %8 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "ReLU 2D", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%7 : memref<512x64xf32>) outs(%8 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %26 = arith.cmpf ogt, %arg8, %cst : f32
      %27 = select %26, %arg8, %cst : f32
      linalg.yield %27 : f32
    }
    memref.dealloc %7 : memref<512x64xf32>
    %9 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%0, %9) : memref<512x64xf32>, memref<512x64xf32> 
    %10 = memref.cast %arg4 : memref<512x512xf32> to memref<?x?xf32, #map0>
    %11 = memref.cast %8 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %12 = memref.cast %9 : memref<512x64xf32> to memref<?x?xf32, #map0>
    call @smatmul(%10, %11, %12) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    memref.dealloc %8 : memref<512x64xf32>
    %13 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel"]} ins(%9, %arg5 : memref<512x64xf32>, memref<512xf32>) outs(%13 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %26 = arith.addf %arg8, %arg9 : f32
      linalg.yield %26 : f32
    }
    memref.dealloc %9 : memref<512x64xf32>
    %14 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "ReLU 2D", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%13 : memref<512x64xf32>) outs(%14 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %26 = arith.cmpf ogt, %arg8, %cst : f32
      %27 = select %26, %arg8, %cst : f32
      linalg.yield %27 : f32
    }
    memref.dealloc %13 : memref<512x64xf32>
    %15 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%1, %15) : memref<10x64xf32>, memref<10x64xf32> 
    %16 = memref.cast %arg6 : memref<10x512xf32> to memref<?x?xf32, #map0>
    %17 = memref.cast %14 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %18 = memref.cast %15 : memref<10x64xf32> to memref<?x?xf32, #map0>
    call @smatmul(%16, %17, %18) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    memref.dealloc %14 : memref<512x64xf32>
    %19 = memref.alloc() : memref<10x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel"]} ins(%15, %arg7 : memref<10x64xf32>, memref<10xf32>) outs(%19 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %26 = arith.addf %arg8, %arg9 : f32
      linalg.yield %26 : f32
    }
    memref.dealloc %15 : memref<10x64xf32>
    %20 = memref.subview %19[0, 0] [1, 64] [1, 1] : memref<10x64xf32> to memref<64xf32, #map1>
    %21 = memref.alloca() : memref<64xf32>
    linalg.copy(%20, %21) : memref<64xf32, #map1>, memref<64xf32> 
    scf.for %arg8 = %c1 to %c10 step %c1 {
      scf.for %arg9 = %c0 to %c64 step %c1 {
        %26 = memref.load %19[%arg8, %arg9] : memref<10x64xf32>
        %27 = memref.load %21[%arg9] : memref<64xf32>
        %28 = arith.cmpf ogt, %26, %27 : f32
        %29 = select %28, %26, %27 : f32
        memref.store %29, %21[%arg9] : memref<64xf32>
      }
    }
    %22 = memref.alloc() : memref<10x64xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%19, %21 : memref<10x64xf32>, memref<64xf32>) outs(%22 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %26 = arith.subf %arg8, %arg9 : f32
      %27 = math.exp %26 : f32
      linalg.yield %27 : f32
    }
    memref.dealloc %19 : memref<10x64xf32>
    %23 = memref.alloca() : memref<64xf32>
    linalg.copy(%2, %23) : memref<64xf32>, memref<64xf32> 
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel"]} ins(%22 : memref<10x64xf32>) outs(%23 : memref<64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %26 = arith.addf %arg8, %arg9 : f32
      linalg.yield %26 : f32
    }
    %24 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %cst) -> (f32) {
      %26 = memref.load %arg1[%arg8] : memref<64xi32>
      %27 = arith.index_cast %26 : i32 to index
      %28 = memref.load %22[%27, %arg8] : memref<10x64xf32>
      %29 = memref.load %23[%arg8] : memref<64xf32>
      %30 = arith.divf %28, %29 : f32
      %31 = math.log %30 : f32
      %32 = arith.negf %31 : f32
      %33 = arith.addf %arg9, %32 : f32
      scf.yield %33 : f32
    }
    memref.dealloc %22 : memref<10x64xf32>
    %25 = arith.divf %24, %cst_0 : f32
    return %25 : f32
  }
  func @__grad_mlir_mlp_batched(%arg0: memref<784x64xf32>, %arg1: memref<64xi32>, %arg2: memref<512x784xf32>, %arg3: memref<512xf32>, %arg4: memref<512x512xf32>, %arg5: memref<512xf32>, %arg6: memref<10x512xf32>, %arg7: memref<10xf32>) -> (memref<512x784xf32>, memref<512xf32>, memref<512x512xf32>, memref<512xf32>, memref<10x512xf32>, memref<10xf32>) {
    %cst = arith.constant 1.562500e-02 : f32
    %c63 = arith.constant 63 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_10xf32 : memref<10xf32>
    %1 = memref.get_global @__constant_512x64xf32 : memref<512x64xf32>
    %2 = memref.get_global @__constant_512xf32 : memref<512xf32>
    %3 = memref.get_global @__constant_10x64xf32 : memref<10x64xf32>
    %4 = memref.get_global @__constant_64xf32 : memref<64xf32>
    %5 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%1, %5) : memref<512x64xf32>, memref<512x64xf32> 
    %6 = memref.cast %arg2 : memref<512x784xf32> to memref<?x?xf32, #map0>
    %7 = memref.cast %arg0 : memref<784x64xf32> to memref<?x?xf32, #map0>
    %8 = memref.cast %5 : memref<512x64xf32> to memref<?x?xf32, #map0>
    call @smatmul(%6, %7, %8) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    %9 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel"]} ins(%5, %arg3 : memref<512x64xf32>, memref<512xf32>) outs(%9 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %5 : memref<512x64xf32>
    %10 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "ReLU 2D", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%9 : memref<512x64xf32>) outs(%10 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.cmpf ogt, %arg8, %cst_0 : f32
      %69 = select %68, %arg8, %cst_0 : f32
      linalg.yield %69 : f32
    }
    %11 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%1, %11) : memref<512x64xf32>, memref<512x64xf32> 
    %12 = memref.cast %arg4 : memref<512x512xf32> to memref<?x?xf32, #map0>
    %13 = memref.cast %10 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %14 = memref.cast %11 : memref<512x64xf32> to memref<?x?xf32, #map0>
    call @smatmul(%12, %13, %14) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    %15 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel"]} ins(%11, %arg5 : memref<512x64xf32>, memref<512xf32>) outs(%15 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %11 : memref<512x64xf32>
    %16 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "ReLU 2D", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%15 : memref<512x64xf32>) outs(%16 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.cmpf ogt, %arg8, %cst_0 : f32
      %69 = select %68, %arg8, %cst_0 : f32
      linalg.yield %69 : f32
    }
    %17 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%3, %17) : memref<10x64xf32>, memref<10x64xf32> 
    %18 = memref.cast %arg6 : memref<10x512xf32> to memref<?x?xf32, #map0>
    %19 = memref.cast %16 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %20 = memref.cast %17 : memref<10x64xf32> to memref<?x?xf32, #map0>
    call @smatmul(%18, %19, %20) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    %21 = memref.alloc() : memref<10x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel"]} ins(%17, %arg7 : memref<10x64xf32>, memref<10xf32>) outs(%21 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %17 : memref<10x64xf32>
    %22 = memref.subview %21[0, 0] [1, 64] [1, 1] : memref<10x64xf32> to memref<64xf32, #map1>
    %23 = memref.alloca() : memref<64xf32>
    linalg.copy(%22, %23) : memref<64xf32, #map1>, memref<64xf32> 
    %24 = memref.alloca() : memref<64x9xi1>
    scf.for %arg8 = %c1 to %c10 step %c1 {
      scf.for %arg9 = %c0 to %c64 step %c1 {
        %68 = memref.load %21[%arg8, %arg9] : memref<10x64xf32>
        %69 = memref.load %23[%arg9] : memref<64xf32>
        %70 = arith.cmpf ogt, %68, %69 : f32
        memref.store %70, %24[%arg9, %arg8] {lagrad_cache} : memref<64x9xi1>
        %71 = select %70, %68, %69 : f32
        memref.store %71, %23[%arg9] : memref<64xf32>
      }
    }
    %25 = memref.alloc() : memref<10x64xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%21, %23 : memref<10x64xf32>, memref<64xf32>) outs(%25 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.subf %arg8, %arg9 : f32
      %69 = math.exp %68 : f32
      linalg.yield %69 : f32
    }
    %26 = memref.alloca() : memref<64xf32>
    linalg.copy(%4, %26) : memref<64xf32>, memref<64xf32> 
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel"]} ins(%25 : memref<10x64xf32>) outs(%26 : memref<64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    %27 = memref.alloca() : memref<64xf32>
    linalg.fill(%cst_0, %27) : f32, memref<64xf32> 
    %28 = memref.alloc() : memref<10x64xf32>
    linalg.fill(%cst_0, %28) : f32, memref<10x64xf32> 
    scf.for %arg8 = %c0 to %c64 step %c1 {
      %68 = arith.subi %c63, %arg8 : index
      %69 = memref.load %arg1[%68] : memref<64xi32>
      %70 = arith.index_cast %69 {"cloned "} : i32 to index
      %71 = memref.load %25[%70, %68] : memref<10x64xf32>
      %72 = memref.load %26[%68] : memref<64xf32>
      %73 = arith.divf %71, %72 {"cloned "} : f32
      %74 = arith.negf %cst : f32
      %75 = arith.divf %74, %73 : f32
      %76 = arith.divf %75, %72 : f32
      %77 = arith.mulf %75, %71 : f32
      %78 = arith.negf %77 : f32
      %79 = arith.mulf %72, %72 : f32
      %80 = arith.divf %78, %79 : f32
      %81 = memref.load %27[%68] : memref<64xf32>
      %82 = arith.addf %81, %80 : f32
      memref.store %82, %27[%68] : memref<64xf32>
      %83 = memref.load %28[%70, %68] : memref<10x64xf32>
      %84 = arith.addf %83, %76 : f32
      memref.store %84, %28[%70, %68] : memref<10x64xf32>
    }
    memref.dealloc %25 : memref<10x64xf32>
    %29 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%3, %29) : memref<10x64xf32>, memref<10x64xf32> 
    linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%27 : memref<64xf32>) outs(%29 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    %30 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%28, %30) : memref<10x64xf32>, memref<10x64xf32> 
    memref.dealloc %28 : memref<10x64xf32>
    linalg.generic {doc = "Add in place", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%29 : memref<10x64xf32>) outs(%30 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %29 : memref<10x64xf32>
    %31 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%3, %31) : memref<10x64xf32>, memref<10x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map3, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%21, %23, %30 : memref<10x64xf32>, memref<64xf32>, memref<10x64xf32>) outs(%31 : memref<10x64xf32>) attrs =  {"adjoint of ", "gradient space for "} {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32, %arg11: f32):  // no predecessors
      %68 = arith.subf %arg8, %arg9 : f32
      %69 = math.exp %68 : f32
      %70 = arith.mulf %arg10, %69 : f32
      %71 = arith.addf %70, %arg11 : f32
      linalg.yield %71 : f32
    }
    %32 = memref.alloca() : memref<64xf32>
    linalg.copy(%4, %32) : memref<64xf32>, memref<64xf32> 
    linalg.generic {indexing_maps = [#map2, #map3, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%21, %23, %30 : memref<10x64xf32>, memref<64xf32>, memref<10x64xf32>) outs(%32 : memref<64xf32>) attrs =  {"adjoint of "} {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32, %arg11: f32):  // no predecessors
      %68 = arith.subf %arg8, %arg9 : f32
      %69 = math.exp %68 : f32
      %70 = arith.mulf %arg10, %69 : f32
      %71 = arith.negf %70 : f32
      %72 = arith.addf %71, %arg11 : f32
      linalg.yield %72 : f32
    }
    memref.dealloc %30 : memref<10x64xf32>
    memref.dealloc %21 : memref<10x64xf32>
    scf.for %arg8 = %c1 to %c10 step %c1 {
      %68 = arith.subi %c10, %arg8 : index
      scf.for %arg9 = %c0 to %c64 step %c1 {
        %69 = arith.subi %c63, %arg9 : index
        %70 = memref.load %24[%69, %68] {"cached "} : memref<64x9xi1>
        %71 = memref.load %32[%69] : memref<64xf32>
        memref.store %cst_0, %32[%69] : memref<64xf32>
        %72 = select %70, %71, %cst_0 : f32
        %73 = select %70, %cst_0, %71 : f32
        %74 = memref.load %32[%69] : memref<64xf32>
        %75 = arith.addf %74, %73 : f32
        memref.store %75, %32[%69] : memref<64xf32>
        %76 = memref.load %31[%68, %69] : memref<10x64xf32>
        %77 = arith.addf %76, %72 : f32
        memref.store %77, %31[%68, %69] : memref<10x64xf32>
      }
    }
    %33 = memref.subview %31[0, 0] [1, 64] [1, 1] : memref<10x64xf32> to memref<64xf32, #map1>
    %34 = memref.cast %33 : memref<64xf32, #map1> to memref<64xf32>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%34, %32 : memref<64xf32>, memref<64xf32>) outs(%33 : memref<64xf32, #map1>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    %35 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%3, %35) : memref<10x64xf32>, memref<10x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%31 : memref<10x64xf32>) outs(%35 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    %36 = memref.alloc() : memref<10xf32>
    linalg.copy(%0, %36) : memref<10xf32>, memref<10xf32> 
    linalg.generic {indexing_maps = [#map2, #map6], iterator_types = ["parallel", "parallel"]} ins(%31 : memref<10x64xf32>) outs(%36 : memref<10xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %31 : memref<10x64xf32>
    %37 = memref.alloc() : memref<10x512xf32>
    linalg.fill(%cst_0, %37) : f32, memref<10x512xf32> 
    %38 = memref.alloc() : memref<10x512xf32>
    linalg.copy(%37, %38) : memref<10x512xf32>, memref<10x512xf32> 
    memref.dealloc %37 : memref<10x512xf32>
    %39 = memref.cast %35 : memref<10x64xf32> to memref<?x?xf32, #map0>
    %40 = memref.cast %16 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %41 = memref.cast %38 : memref<10x512xf32> to memref<?x?xf32, #map0>
    call @smatmul_grad_first(%39, %40, %41) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    memref.dealloc %16 : memref<512x64xf32>
    %42 = memref.alloc() : memref<512x64xf32>
    linalg.fill(%cst_0, %42) : f32, memref<512x64xf32> 
    %43 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%42, %43) : memref<512x64xf32>, memref<512x64xf32> 
    memref.dealloc %42 : memref<512x64xf32>
    %44 = memref.cast %arg6 : memref<10x512xf32> to memref<?x?xf32, #map0>
    %45 = memref.cast %35 : memref<10x64xf32> to memref<?x?xf32, #map0>
    %46 = memref.cast %43 : memref<512x64xf32> to memref<?x?xf32, #map0>
    call @smatmul_grad_second(%44, %45, %46) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    memref.dealloc %35 : memref<10x64xf32>
    %47 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%1, %47) : memref<512x64xf32>, memref<512x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%15, %43 : memref<512x64xf32>, memref<512x64xf32>) outs(%47 : memref<512x64xf32>) attrs =  {"adjoint of %h1"} {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.cmpf ogt, %arg8, %cst_0 : f32
      %69 = select %68, %arg9, %cst_0 : f32
      %70 = arith.addf %69, %arg10 : f32
      linalg.yield %70 : f32
    }
    memref.dealloc %43 : memref<512x64xf32>
    memref.dealloc %15 : memref<512x64xf32>
    %48 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%1, %48) : memref<512x64xf32>, memref<512x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%47 : memref<512x64xf32>) outs(%48 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    %49 = memref.alloc() : memref<512xf32>
    linalg.copy(%2, %49) : memref<512xf32>, memref<512xf32> 
    linalg.generic {indexing_maps = [#map2, #map6], iterator_types = ["parallel", "parallel"]} ins(%47 : memref<512x64xf32>) outs(%49 : memref<512xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %47 : memref<512x64xf32>
    %50 = memref.alloc() : memref<512x512xf32>
    linalg.fill(%cst_0, %50) : f32, memref<512x512xf32> 
    %51 = memref.alloc() : memref<512x512xf32>
    linalg.copy(%50, %51) : memref<512x512xf32>, memref<512x512xf32> 
    memref.dealloc %50 : memref<512x512xf32>
    %52 = memref.cast %48 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %53 = memref.cast %10 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %54 = memref.cast %51 : memref<512x512xf32> to memref<?x?xf32, #map0>
    call @smatmul_grad_first(%52, %53, %54) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    memref.dealloc %10 : memref<512x64xf32>
    %55 = memref.alloc() : memref<512x64xf32>
    linalg.fill(%cst_0, %55) : f32, memref<512x64xf32> 
    %56 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%55, %56) : memref<512x64xf32>, memref<512x64xf32> 
    memref.dealloc %55 : memref<512x64xf32>
    %57 = memref.cast %arg4 : memref<512x512xf32> to memref<?x?xf32, #map0>
    %58 = memref.cast %48 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %59 = memref.cast %56 : memref<512x64xf32> to memref<?x?xf32, #map0>
    call @smatmul_grad_second(%57, %58, %59) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    memref.dealloc %48 : memref<512x64xf32>
    %60 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%1, %60) : memref<512x64xf32>, memref<512x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%9, %56 : memref<512x64xf32>, memref<512x64xf32>) outs(%60 : memref<512x64xf32>) attrs =  {"adjoint of %h0"} {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.cmpf ogt, %arg8, %cst_0 : f32
      %69 = select %68, %arg9, %cst_0 : f32
      %70 = arith.addf %69, %arg10 : f32
      linalg.yield %70 : f32
    }
    memref.dealloc %56 : memref<512x64xf32>
    memref.dealloc %9 : memref<512x64xf32>
    %61 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%1, %61) : memref<512x64xf32>, memref<512x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%60 : memref<512x64xf32>) outs(%61 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    %62 = memref.alloc() : memref<512xf32>
    linalg.copy(%2, %62) : memref<512xf32>, memref<512xf32> 
    linalg.generic {indexing_maps = [#map2, #map6], iterator_types = ["parallel", "parallel"]} ins(%60 : memref<512x64xf32>) outs(%62 : memref<512xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %60 : memref<512x64xf32>
    %63 = memref.alloc() : memref<512x784xf32>
    linalg.fill(%cst_0, %63) : f32, memref<512x784xf32> 
    %64 = memref.alloc() : memref<512x784xf32>
    linalg.copy(%63, %64) : memref<512x784xf32>, memref<512x784xf32> 
    memref.dealloc %63 : memref<512x784xf32>
    %65 = memref.cast %61 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %66 = memref.cast %arg0 : memref<784x64xf32> to memref<?x?xf32, #map0>
    %67 = memref.cast %64 : memref<512x784xf32> to memref<?x?xf32, #map0>
    call @smatmul_grad_first(%65, %66, %67) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    memref.dealloc %61 : memref<512x64xf32>
    return %64, %62, %51, %49, %38, %36 : memref<512x784xf32>, memref<512xf32>, memref<512x512xf32>, memref<512xf32>, memref<10x512xf32>, memref<10xf32>
  }
  func @mlir_mlp(%arg0: memref<64x784xf32>, %arg1: memref<64xi32>, %arg2: memref<512x784xf32>, %arg3: memref<512xf32>, %arg4: memref<512x512xf32>, %arg5: memref<512xf32>, %arg6: memref<10x512xf32>, %arg7: memref<10xf32>) -> f32 {
    %c10 = arith.constant 10 : index
    %cst = arith.constant 6.400000e+01 : f32
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_512xf32 : memref<512xf32>
    %1 = memref.get_global @__constant_10xf32 : memref<10xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<512xf32>
    %6 = memref.alloc() : memref<512xf32>
    %7 = memref.alloc() : memref<512xf32>
    %8 = memref.alloca() : memref<10xf32>
    %9 = memref.alloca() : memref<10xf32>
    %10 = memref.alloca() : memref<10xf32>
    %11 = memref.alloca() : memref<10xf32>
    %12 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %cst_0) -> (f32) {
      %14 = memref.subview %arg0[%arg8, 0] [1, 784] [1, 1] : memref<64x784xf32> to memref<784xf32, #map1>
      linalg.copy(%0, %2) : memref<512xf32>, memref<512xf32> 
      %15 = memref.cast %arg2 : memref<512x784xf32> to memref<?x?xf32, #map0>
      %16 = memref.cast %14 : memref<784xf32, #map1> to memref<?xf32, #map1>
      %17 = memref.cast %2 : memref<512xf32> to memref<?xf32, #map1>
      call @smatvec(%15, %16, %17) : (memref<?x?xf32, #map0>, memref<?xf32, #map1>, memref<?xf32, #map1>) -> ()
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%2, %arg3 : memref<512xf32>, memref<512xf32>) outs(%3 : memref<512xf32>) {
      ^bb0(%arg10: f32, %arg11: f32, %arg12: f32):  // no predecessors
        %33 = arith.addf %arg10, %arg11 : f32
        linalg.yield %33 : f32
      }
      linalg.generic {doc = "ReLU", indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%3 : memref<512xf32>) outs(%4 : memref<512xf32>) {
      ^bb0(%arg10: f32, %arg11: f32):  // no predecessors
        %33 = arith.cmpf ogt, %arg10, %cst_0 : f32
        %34 = select %33, %arg10, %cst_0 : f32
        linalg.yield %34 : f32
      }
      linalg.copy(%0, %5) : memref<512xf32>, memref<512xf32> 
      %18 = memref.cast %arg4 : memref<512x512xf32> to memref<?x?xf32, #map0>
      %19 = memref.cast %4 : memref<512xf32> to memref<?xf32, #map1>
      %20 = memref.cast %5 : memref<512xf32> to memref<?xf32, #map1>
      call @smatvec(%18, %19, %20) : (memref<?x?xf32, #map0>, memref<?xf32, #map1>, memref<?xf32, #map1>) -> ()
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%5, %arg5 : memref<512xf32>, memref<512xf32>) outs(%6 : memref<512xf32>) {
      ^bb0(%arg10: f32, %arg11: f32, %arg12: f32):  // no predecessors
        %33 = arith.addf %arg10, %arg11 : f32
        linalg.yield %33 : f32
      }
      linalg.generic {doc = "ReLU", indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%6 : memref<512xf32>) outs(%7 : memref<512xf32>) {
      ^bb0(%arg10: f32, %arg11: f32):  // no predecessors
        %33 = arith.cmpf ogt, %arg10, %cst_0 : f32
        %34 = select %33, %arg10, %cst_0 : f32
        linalg.yield %34 : f32
      }
      linalg.copy(%1, %8) : memref<10xf32>, memref<10xf32> 
      %21 = memref.cast %arg6 : memref<10x512xf32> to memref<?x?xf32, #map0>
      %22 = memref.cast %7 : memref<512xf32> to memref<?xf32, #map1>
      %23 = memref.cast %8 : memref<10xf32> to memref<?xf32, #map1>
      call @smatvec(%21, %22, %23) : (memref<?x?xf32, #map0>, memref<?xf32, #map1>, memref<?xf32, #map1>) -> ()
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%8, %arg7 : memref<10xf32>, memref<10xf32>) outs(%9 : memref<10xf32>) {
      ^bb0(%arg10: f32, %arg11: f32, %arg12: f32):  // no predecessors
        %33 = arith.addf %arg10, %arg11 : f32
        linalg.yield %33 : f32
      }
      %24 = memref.load %arg1[%arg8] : memref<64xi32>
      %25 = arith.index_cast %24 : i32 to index
      %26 = memref.load %9[%c0] : memref<10xf32>
      %27 = scf.for %arg10 = %c1 to %c10 step %c1 iter_args(%arg11 = %26) -> (f32) {
        %33 = memref.load %9[%arg10] : memref<10xf32>
        %34 = arith.cmpf ogt, %33, %arg11 : f32
        %35 = select %34, %33, %arg11 : f32
        scf.yield %35 : f32
      }
      linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%9 : memref<10xf32>) outs(%10 : memref<10xf32>) {
      ^bb0(%arg10: f32, %arg11: f32):  // no predecessors
        %33 = arith.subf %arg10, %27 : f32
        %34 = math.exp %33 : f32
        linalg.yield %34 : f32
      }
      %28 = scf.for %arg10 = %c0 to %c10 step %c1 iter_args(%arg11 = %cst_0) -> (f32) {
        %33 = memref.load %10[%arg10] : memref<10xf32>
        %34 = arith.addf %arg11, %33 : f32
        scf.yield %34 : f32
      }
      linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%10 : memref<10xf32>) outs(%11 : memref<10xf32>) {
      ^bb0(%arg10: f32, %arg11: f32):  // no predecessors
        %33 = arith.divf %arg10, %28 : f32
        linalg.yield %33 : f32
      }
      %29 = memref.load %11[%25] : memref<10xf32>
      %30 = math.log %29 : f32
      %31 = arith.negf %30 : f32
      %32 = arith.addf %arg9, %31 : f32
      scf.yield %32 : f32
    }
    memref.dealloc %7 : memref<512xf32>
    memref.dealloc %6 : memref<512xf32>
    memref.dealloc %5 : memref<512xf32>
    memref.dealloc %4 : memref<512xf32>
    memref.dealloc %3 : memref<512xf32>
    memref.dealloc %2 : memref<512xf32>
    %13 = arith.divf %12, %cst : f32
    return %13 : f32
  }
  func @__grad_mlir_mlp(%arg0: memref<64x784xf32>, %arg1: memref<64xi32>, %arg2: memref<512x784xf32>, %arg3: memref<512xf32>, %arg4: memref<512x512xf32>, %arg5: memref<512xf32>, %arg6: memref<10x512xf32>, %arg7: memref<10xf32>) -> (memref<512x784xf32>, memref<512xf32>, memref<512x512xf32>, memref<512xf32>, memref<10x512xf32>, memref<10xf32>) {
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c63 = arith.constant 63 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.562500e-02 : f32
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_512xf32 : memref<512xf32>
    %1 = memref.get_global @__constant_10xf32 : memref<10xf32>
    %2 = memref.get_global @__constant_xf32 : memref<f32>
    %3 = memref.alloc() : memref<10x512xf32>
    linalg.fill(%cst, %3) : f32, memref<10x512xf32> 
    %4 = memref.alloc() : memref<512xf32>
    linalg.fill(%cst, %4) : f32, memref<512xf32> 
    %5 = memref.alloc() : memref<512x784xf32>
    linalg.fill(%cst, %5) : f32, memref<512x784xf32> 
    %6 = memref.alloc() : memref<512x512xf32>
    linalg.fill(%cst, %6) : f32, memref<512x512xf32> 
    %7 = memref.alloca() : memref<10xf32>
    linalg.fill(%cst, %7) : f32, memref<10xf32> 
    %8 = memref.alloc() : memref<512xf32>
    linalg.fill(%cst, %8) : f32, memref<512xf32> 
    %9 = memref.alloc() : memref<512xf32>
    %10 = memref.alloc() : memref<512xf32>
    %11 = memref.alloc() : memref<512xf32>
    %12 = memref.alloc() : memref<512xf32>
    %13 = memref.alloc() : memref<512xf32>
    %14 = memref.alloc() : memref<512xf32>
    %15 = memref.alloca() : memref<10xf32>
    %16 = memref.alloca() : memref<10xf32>
    %17 = memref.alloca() : memref<9xi1>
    %18 = memref.alloca() : memref<10xf32>
    %19 = memref.alloca() : memref<10xf32>
    %20 = memref.alloca() : memref<10xf32>
    %21 = memref.alloca() : memref<f32>
    %22 = memref.alloca() : memref<10xf32>
    %23 = memref.alloca() : memref<f32>
    %24 = memref.alloca() : memref<10xf32>
    %25 = memref.alloc() : memref<10x512xf32>
    %26 = memref.alloc() : memref<512xf32>
    %27 = memref.alloc() : memref<512xf32>
    %28 = memref.alloc() : memref<512x512xf32>
    %29 = memref.alloc() : memref<512xf32>
    %30 = memref.alloc() : memref<512xf32>
    %31 = memref.alloc() : memref<512x784xf32>
    %32 = memref.alloc() : memref<10xf32>
    %33 = memref.alloc() : memref<10x512xf32>
    %34 = memref.alloc() : memref<512xf32>
    %35 = memref.alloc() : memref<512x512xf32>
    %36 = memref.alloc() : memref<512xf32>
    %37 = memref.alloc() : memref<512x784xf32>
    %38:6 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %3, %arg10 = %4, %arg11 = %5, %arg12 = %6, %arg13 = %7, %arg14 = %8) -> (memref<10x512xf32>, memref<512xf32>, memref<512x784xf32>, memref<512x512xf32>, memref<10xf32>, memref<512xf32>) {
      %39 = arith.subi %c63, %arg8 : index
      %40 = memref.subview %arg0[%39, 0] [1, 784] [1, 1] : memref<64x784xf32> to memref<784xf32, #map1>
      linalg.copy(%0, %9) : memref<512xf32>, memref<512xf32> 
      %41 = memref.cast %arg2 : memref<512x784xf32> to memref<?x?xf32, #map0>
      %42 = memref.cast %40 : memref<784xf32, #map1> to memref<?xf32, #map1>
      %43 = memref.cast %9 : memref<512xf32> to memref<?xf32, #map1>
      call @smatvec(%41, %42, %43) : (memref<?x?xf32, #map0>, memref<?xf32, #map1>, memref<?xf32, #map1>) -> ()
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%9, %arg3 : memref<512xf32>, memref<512xf32>) outs(%10 : memref<512xf32>) {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 {"cloned %h0_1"} : f32
        linalg.yield %78 : f32
      }
      linalg.generic {doc = "ReLU", indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%10 : memref<512xf32>) outs(%11 : memref<512xf32>) attrs =  {"cloned %h0"} {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.cmpf ogt, %arg15, %cst : f32
        %79 = select %78, %arg15, %cst : f32
        linalg.yield %79 : f32
      }
      linalg.copy(%0, %12) : memref<512xf32>, memref<512xf32> 
      %44 = memref.cast %arg4 : memref<512x512xf32> to memref<?x?xf32, #map0>
      %45 = memref.cast %11 : memref<512xf32> to memref<?xf32, #map1>
      %46 = memref.cast %12 : memref<512xf32> to memref<?xf32, #map1>
      call @smatvec(%44, %45, %46) : (memref<?x?xf32, #map0>, memref<?xf32, #map1>, memref<?xf32, #map1>) -> ()
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%12, %arg5 : memref<512xf32>, memref<512xf32>) outs(%13 : memref<512xf32>) {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 {"cloned %h1_1"} : f32
        linalg.yield %78 : f32
      }
      linalg.generic {doc = "ReLU", indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%13 : memref<512xf32>) outs(%14 : memref<512xf32>) attrs =  {"cloned %h1"} {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.cmpf ogt, %arg15, %cst : f32
        %79 = select %78, %arg15, %cst : f32
        linalg.yield %79 : f32
      }
      linalg.copy(%1, %15) : memref<10xf32>, memref<10xf32> 
      %47 = memref.cast %arg6 : memref<10x512xf32> to memref<?x?xf32, #map0>
      %48 = memref.cast %14 : memref<512xf32> to memref<?xf32, #map1>
      %49 = memref.cast %15 : memref<10xf32> to memref<?xf32, #map1>
      call @smatvec(%47, %48, %49) : (memref<?x?xf32, #map0>, memref<?xf32, #map1>, memref<?xf32, #map1>) -> ()
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%15, %arg7 : memref<10xf32>, memref<10xf32>) outs(%16 : memref<10xf32>) {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 {"cloned %activations"} : f32
        linalg.yield %78 : f32
      }
      %50 = memref.load %arg1[%39] : memref<64xi32>
      %51 = arith.index_cast %50 {"cloned %l_idx"} : i32 to index
      %52 = memref.load %16[%c0] : memref<10xf32>
      %53 = scf.for %arg15 = %c1 to %c10 step %c1 iter_args(%arg16 = %52) -> (f32) {
        %78 = memref.load %16[%arg15] : memref<10xf32>
        %79 = arith.cmpf ogt, %78, %arg16 : f32
        memref.store %79, %17[%arg15] {lagrad_cache} : memref<9xi1>
        %80 = select %79, %78, %arg16 : f32
        scf.yield %80 : f32
      }
      linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%16 : memref<10xf32>) outs(%18 : memref<10xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.subf %arg15, %53 : f32
        %79 = math.exp %78 : f32
        linalg.yield %79 : f32
      }
      %54 = scf.for %arg15 = %c0 to %c10 step %c1 iter_args(%arg16 = %cst) -> (f32) {
        %78 = memref.load %18[%arg15] : memref<10xf32>
        %79 = arith.addf %arg16, %78 : f32
        scf.yield %79 : f32
      }
      linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%18 : memref<10xf32>) outs(%19 : memref<10xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.divf %arg15, %54 : f32
        linalg.yield %78 : f32
      }
      %55 = memref.load %19[%51] : memref<10xf32>
      %56 = arith.negf %cst_0 : f32
      %57 = arith.divf %56, %55 : f32
      linalg.fill(%cst, %20) : f32, memref<10xf32> 
      memref.store %57, %20[%51] : memref<10xf32>
      linalg.copy(%2, %21) : memref<f32>, memref<f32> 
      linalg.generic {indexing_maps = [#map4, #map4, #map5], iterator_types = ["parallel"]} ins(%18, %20 : memref<10xf32>, memref<10xf32>) outs(%21 : memref<f32>) attrs =  {"adjoint of "} {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.mulf %arg16, %arg15 : f32
        %79 = arith.negf %78 : f32
        %80 = arith.mulf %54, %54 : f32
        %81 = arith.divf %79, %80 : f32
        %82 = arith.addf %81, %arg17 : f32
        linalg.yield %82 : f32
      }
      %58 = memref.load %21[] : memref<f32>
      linalg.copy(%1, %22) : memref<10xf32>, memref<10xf32> 
      linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%20 : memref<10xf32>) outs(%22 : memref<10xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.divf %arg15, %54 : f32
        %79 = arith.addf %78, %arg16 : f32
        linalg.yield %79 : f32
      }
      scf.for %arg15 = %c0 to %c10 step %c1 {
        %78 = arith.subi %c9, %arg15 : index
        %79 = memref.load %22[%78] : memref<10xf32>
        %80 = arith.addf %79, %58 : f32
        memref.store %80, %22[%78] : memref<10xf32>
      }
      linalg.copy(%2, %23) : memref<f32>, memref<f32> 
      linalg.generic {indexing_maps = [#map4, #map4, #map5], iterator_types = ["parallel"]} ins(%16, %22 : memref<10xf32>, memref<10xf32>) outs(%23 : memref<f32>) attrs =  {"adjoint of "} {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.subf %arg15, %53 : f32
        %79 = math.exp %78 : f32
        %80 = arith.mulf %arg16, %79 : f32
        %81 = arith.negf %80 : f32
        %82 = arith.addf %81, %arg17 : f32
        linalg.yield %82 : f32
      }
      %59 = memref.load %23[] : memref<f32>
      linalg.copy(%1, %24) : memref<10xf32>, memref<10xf32> 
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%16, %22 : memref<10xf32>, memref<10xf32>) outs(%24 : memref<10xf32>) attrs =  {"adjoint of ", "gradient space for "} {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.subf %arg15, %53 : f32
        %79 = math.exp %78 : f32
        %80 = arith.mulf %arg16, %79 : f32
        %81 = arith.addf %80, %arg17 : f32
        linalg.yield %81 : f32
      }
      %60 = scf.for %arg15 = %c1 to %c10 step %c1 iter_args(%arg16 = %59) -> (f32) {
        %78 = arith.subi %c10, %arg15 : index
        %79 = memref.load %17[%78] {"cached "} : memref<9xi1>
        %80 = select %79, %arg16, %cst : f32
        %81 = select %79, %cst, %arg16 : f32
        %82 = memref.load %24[%78] : memref<10xf32>
        %83 = arith.addf %82, %80 : f32
        memref.store %83, %24[%78] : memref<10xf32>
        scf.yield %81 : f32
      }
      %61 = memref.load %24[%c0] : memref<10xf32>
      %62 = arith.addf %61, %60 : f32
      memref.store %62, %24[%c0] : memref<10xf32>
      linalg.copy(%arg13, %32) : memref<10xf32>, memref<10xf32> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%24 : memref<10xf32>) outs(%32 : memref<10xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 : f32
        linalg.yield %78 : f32
      }
      %63 = memref.cast %24 : memref<10xf32> to memref<?xf32, #map1>
      %64 = memref.cast %14 : memref<512xf32> to memref<?xf32, #map1>
      %65 = memref.cast %25 : memref<10x512xf32> to memref<?x?xf32, #map0>
      call @souter(%63, %64, %65) : (memref<?xf32, #map1>, memref<?xf32, #map1>, memref<?x?xf32, #map0>) -> ()
      linalg.copy(%arg9, %33) : memref<10x512xf32>, memref<10x512xf32> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%25 : memref<10x512xf32>) outs(%33 : memref<10x512xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 : f32
        linalg.yield %78 : f32
      }
      linalg.copy(%0, %26) : memref<512xf32>, memref<512xf32> 
      %66 = memref.cast %24 : memref<10xf32> to memref<?xf32, #map1>
      %67 = memref.cast %arg6 : memref<10x512xf32> to memref<?x?xf32, #map0>
      %68 = memref.cast %26 : memref<512xf32> to memref<?xf32, #map1>
      call @svecmat(%66, %67, %68) : (memref<?xf32, #map1>, memref<?x?xf32, #map0>, memref<?xf32, #map1>) -> ()
      linalg.copy(%0, %27) : memref<512xf32>, memref<512xf32> 
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%13, %26 : memref<512xf32>, memref<512xf32>) outs(%27 : memref<512xf32>) attrs =  {"adjoint of %h1"} {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.cmpf ogt, %arg15, %cst : f32
        %79 = select %78, %arg16, %cst : f32
        %80 = arith.addf %79, %arg17 : f32
        linalg.yield %80 : f32
      }
      linalg.copy(%arg10, %34) : memref<512xf32>, memref<512xf32> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%27 : memref<512xf32>) outs(%34 : memref<512xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 : f32
        linalg.yield %78 : f32
      }
      %69 = memref.cast %27 : memref<512xf32> to memref<?xf32, #map1>
      %70 = memref.cast %11 : memref<512xf32> to memref<?xf32, #map1>
      %71 = memref.cast %28 : memref<512x512xf32> to memref<?x?xf32, #map0>
      call @souter(%69, %70, %71) : (memref<?xf32, #map1>, memref<?xf32, #map1>, memref<?x?xf32, #map0>) -> ()
      linalg.copy(%arg12, %35) : memref<512x512xf32>, memref<512x512xf32> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%28 : memref<512x512xf32>) outs(%35 : memref<512x512xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 : f32
        linalg.yield %78 : f32
      }
      linalg.copy(%0, %29) : memref<512xf32>, memref<512xf32> 
      %72 = memref.cast %27 : memref<512xf32> to memref<?xf32, #map1>
      %73 = memref.cast %arg4 : memref<512x512xf32> to memref<?x?xf32, #map0>
      %74 = memref.cast %29 : memref<512xf32> to memref<?xf32, #map1>
      call @svecmat(%72, %73, %74) : (memref<?xf32, #map1>, memref<?x?xf32, #map0>, memref<?xf32, #map1>) -> ()
      linalg.copy(%0, %30) : memref<512xf32>, memref<512xf32> 
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%10, %29 : memref<512xf32>, memref<512xf32>) outs(%30 : memref<512xf32>) attrs =  {"adjoint of %h0"} {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.cmpf ogt, %arg15, %cst : f32
        %79 = select %78, %arg16, %cst : f32
        %80 = arith.addf %79, %arg17 : f32
        linalg.yield %80 : f32
      }
      linalg.copy(%arg14, %36) : memref<512xf32>, memref<512xf32> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%30 : memref<512xf32>) outs(%36 : memref<512xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 : f32
        linalg.yield %78 : f32
      }
      %75 = memref.cast %30 : memref<512xf32> to memref<?xf32, #map1>
      %76 = memref.cast %40 : memref<784xf32, #map1> to memref<?xf32, #map1>
      %77 = memref.cast %31 : memref<512x784xf32> to memref<?x?xf32, #map0>
      call @souter(%75, %76, %77) : (memref<?xf32, #map1>, memref<?xf32, #map1>, memref<?x?xf32, #map0>) -> ()
      linalg.copy(%arg11, %37) : memref<512x784xf32>, memref<512x784xf32> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%31 : memref<512x784xf32>) outs(%37 : memref<512x784xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 : f32
        linalg.yield %78 : f32
      }
      scf.yield %33, %34, %37, %35, %32, %36 : memref<10x512xf32>, memref<512xf32>, memref<512x784xf32>, memref<512x512xf32>, memref<10xf32>, memref<512xf32>
    }
    memref.dealloc %31 : memref<512x784xf32>
    memref.dealloc %30 : memref<512xf32>
    memref.dealloc %29 : memref<512xf32>
    memref.dealloc %28 : memref<512x512xf32>
    memref.dealloc %27 : memref<512xf32>
    memref.dealloc %26 : memref<512xf32>
    memref.dealloc %25 : memref<10x512xf32>
    memref.dealloc %14 : memref<512xf32>
    memref.dealloc %13 : memref<512xf32>
    memref.dealloc %12 : memref<512xf32>
    memref.dealloc %11 : memref<512xf32>
    memref.dealloc %10 : memref<512xf32>
    memref.dealloc %9 : memref<512xf32>
    memref.dealloc %8 : memref<512xf32>
    memref.dealloc %6 : memref<512x512xf32>
    memref.dealloc %5 : memref<512x784xf32>
    memref.dealloc %4 : memref<512xf32>
    memref.dealloc %3 : memref<10x512xf32>
    return %38#2, %38#5, %38#3, %38#1, %38#0, %38#4 : memref<512x784xf32>, memref<512xf32>, memref<512x512xf32>, memref<512xf32>, memref<10x512xf32>, memref<10xf32>
  }
  func @lagrad_mlp(%arg0: memref<64x784xf32>, %arg1: memref<64xi32>, %arg2: memref<512x784xf32>, %arg3: memref<512xf32>, %arg4: memref<512x512xf32>, %arg5: memref<512xf32>, %arg6: memref<10x512xf32>, %arg7: memref<10xf32>) -> (memref<512x784xf32>, memref<512xf32>, memref<512x512xf32>, memref<512xf32>, memref<10x512xf32>, memref<10xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 1.562500e-02 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c63 = arith.constant 63 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %0 = memref.get_global @__constant_xf32 : memref<f32>
    %1 = memref.get_global @__constant_10xf32 : memref<10xf32>
    %2 = memref.get_global @__constant_512xf32 : memref<512xf32>
    %3 = memref.alloc() : memref<10x512xf32>
    linalg.fill(%cst_0, %3) : f32, memref<10x512xf32> 
    %4 = memref.alloc() : memref<512xf32>
    linalg.fill(%cst_0, %4) : f32, memref<512xf32> 
    %5 = memref.alloc() : memref<512x784xf32>
    linalg.fill(%cst_0, %5) : f32, memref<512x784xf32> 
    %6 = memref.alloc() : memref<512x512xf32>
    linalg.fill(%cst_0, %6) : f32, memref<512x512xf32> 
    %7 = memref.alloca() : memref<10xf32>
    linalg.fill(%cst_0, %7) : f32, memref<10xf32> 
    %8 = memref.alloc() : memref<512xf32>
    linalg.fill(%cst_0, %8) : f32, memref<512xf32> 
    %9 = memref.alloc() : memref<512xf32>
    %10 = memref.alloc() : memref<512xf32>
    %11 = memref.alloc() : memref<512xf32>
    %12 = memref.alloc() : memref<512xf32>
    %13 = memref.alloc() : memref<512xf32>
    %14 = memref.alloc() : memref<512xf32>
    %15 = memref.alloca() : memref<10xf32>
    %16 = memref.alloca() : memref<10xf32>
    %17 = memref.alloca() : memref<9xi1>
    %18 = memref.alloca() : memref<10xf32>
    %19 = memref.alloca() : memref<10xf32>
    %20 = memref.alloca() : memref<10xf32>
    %21 = memref.alloca() : memref<f32>
    %22 = memref.alloca() : memref<10xf32>
    %23 = memref.alloca() : memref<f32>
    %24 = memref.alloca() : memref<10xf32>
    %25 = memref.alloc() : memref<10x512xf32>
    %26 = memref.alloc() : memref<512xf32>
    %27 = memref.alloc() : memref<512xf32>
    %28 = memref.alloc() : memref<512x512xf32>
    %29 = memref.alloc() : memref<512xf32>
    %30 = memref.alloc() : memref<512xf32>
    %31 = memref.alloc() : memref<512x784xf32>
    %32 = memref.alloc() : memref<10xf32>
    %33 = memref.alloc() : memref<10x512xf32>
    %34 = memref.alloc() : memref<512xf32>
    %35 = memref.alloc() : memref<512x512xf32>
    %36 = memref.alloc() : memref<512xf32>
    %37 = memref.alloc() : memref<512x784xf32>
    %38:6 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %3, %arg10 = %4, %arg11 = %5, %arg12 = %6, %arg13 = %7, %arg14 = %8) -> (memref<10x512xf32>, memref<512xf32>, memref<512x784xf32>, memref<512x512xf32>, memref<10xf32>, memref<512xf32>) {
      %39 = arith.subi %c63, %arg8 : index
      %40 = memref.subview %arg0[%39, 0] [1, 784] [1, 1] : memref<64x784xf32> to memref<784xf32, #map1>
      linalg.copy(%2, %9) : memref<512xf32>, memref<512xf32> 
      %41 = memref.cast %arg2 : memref<512x784xf32> to memref<?x?xf32, #map0>
      %42 = memref.cast %40 : memref<784xf32, #map1> to memref<?xf32, #map1>
      %43 = memref.cast %9 : memref<512xf32> to memref<?xf32, #map1>
      call @smatvec(%41, %42, %43) : (memref<?x?xf32, #map0>, memref<?xf32, #map1>, memref<?xf32, #map1>) -> ()
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%9, %arg3 : memref<512xf32>, memref<512xf32>) outs(%10 : memref<512xf32>) {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 {"cloned %h0_1"} : f32
        linalg.yield %78 : f32
      }
      linalg.generic {doc = "ReLU", indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%10 : memref<512xf32>) outs(%11 : memref<512xf32>) attrs =  {"cloned %h0"} {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.cmpf ogt, %arg15, %cst_0 : f32
        %79 = select %78, %arg15, %cst_0 : f32
        linalg.yield %79 : f32
      }
      linalg.copy(%2, %12) : memref<512xf32>, memref<512xf32> 
      %44 = memref.cast %arg4 : memref<512x512xf32> to memref<?x?xf32, #map0>
      %45 = memref.cast %11 : memref<512xf32> to memref<?xf32, #map1>
      %46 = memref.cast %12 : memref<512xf32> to memref<?xf32, #map1>
      call @smatvec(%44, %45, %46) : (memref<?x?xf32, #map0>, memref<?xf32, #map1>, memref<?xf32, #map1>) -> ()
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%12, %arg5 : memref<512xf32>, memref<512xf32>) outs(%13 : memref<512xf32>) {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 {"cloned %h1_1"} : f32
        linalg.yield %78 : f32
      }
      linalg.generic {doc = "ReLU", indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%13 : memref<512xf32>) outs(%14 : memref<512xf32>) attrs =  {"cloned %h1"} {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.cmpf ogt, %arg15, %cst_0 : f32
        %79 = select %78, %arg15, %cst_0 : f32
        linalg.yield %79 : f32
      }
      linalg.copy(%1, %15) : memref<10xf32>, memref<10xf32> 
      %47 = memref.cast %arg6 : memref<10x512xf32> to memref<?x?xf32, #map0>
      %48 = memref.cast %14 : memref<512xf32> to memref<?xf32, #map1>
      %49 = memref.cast %15 : memref<10xf32> to memref<?xf32, #map1>
      call @smatvec(%47, %48, %49) : (memref<?x?xf32, #map0>, memref<?xf32, #map1>, memref<?xf32, #map1>) -> ()
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%15, %arg7 : memref<10xf32>, memref<10xf32>) outs(%16 : memref<10xf32>) {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 {"cloned %activations"} : f32
        linalg.yield %78 : f32
      }
      %50 = memref.load %arg1[%39] : memref<64xi32>
      %51 = arith.index_cast %50 {"cloned %l_idx"} : i32 to index
      %52 = memref.load %16[%c0] : memref<10xf32>
      %53 = scf.for %arg15 = %c1 to %c10 step %c1 iter_args(%arg16 = %52) -> (f32) {
        %78 = memref.load %16[%arg15] : memref<10xf32>
        %79 = arith.cmpf ogt, %78, %arg16 : f32
        memref.store %79, %17[%arg15] {lagrad_cache} : memref<9xi1>
        %80 = select %79, %78, %arg16 : f32
        scf.yield %80 : f32
      }
      linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%16 : memref<10xf32>) outs(%18 : memref<10xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.subf %arg15, %53 : f32
        %79 = math.exp %78 : f32
        linalg.yield %79 : f32
      }
      %54 = scf.for %arg15 = %c0 to %c10 step %c1 iter_args(%arg16 = %cst_0) -> (f32) {
        %78 = memref.load %18[%arg15] : memref<10xf32>
        %79 = arith.addf %arg16, %78 : f32
        scf.yield %79 : f32
      }
      linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%18 : memref<10xf32>) outs(%19 : memref<10xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.divf %arg15, %54 : f32
        linalg.yield %78 : f32
      }
      %55 = memref.load %19[%51] : memref<10xf32>
      %56 = arith.negf %cst : f32
      %57 = arith.divf %56, %55 : f32
      linalg.fill(%cst_0, %20) : f32, memref<10xf32> 
      memref.store %57, %20[%51] : memref<10xf32>
      linalg.copy(%0, %21) : memref<f32>, memref<f32> 
      linalg.generic {indexing_maps = [#map4, #map4, #map5], iterator_types = ["parallel"]} ins(%18, %20 : memref<10xf32>, memref<10xf32>) outs(%21 : memref<f32>) attrs =  {"adjoint of "} {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.mulf %arg16, %arg15 : f32
        %79 = arith.negf %78 : f32
        %80 = arith.mulf %54, %54 : f32
        %81 = arith.divf %79, %80 : f32
        %82 = arith.addf %81, %arg17 : f32
        linalg.yield %82 : f32
      }
      %58 = memref.load %21[] : memref<f32>
      linalg.copy(%1, %22) : memref<10xf32>, memref<10xf32> 
      linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%20 : memref<10xf32>) outs(%22 : memref<10xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.divf %arg15, %54 : f32
        %79 = arith.addf %78, %arg16 : f32
        linalg.yield %79 : f32
      }
      scf.for %arg15 = %c0 to %c10 step %c1 {
        %78 = arith.subi %c9, %arg15 : index
        %79 = memref.load %22[%78] : memref<10xf32>
        %80 = arith.addf %79, %58 : f32
        memref.store %80, %22[%78] : memref<10xf32>
      }
      linalg.copy(%0, %23) : memref<f32>, memref<f32> 
      linalg.generic {indexing_maps = [#map4, #map4, #map5], iterator_types = ["parallel"]} ins(%16, %22 : memref<10xf32>, memref<10xf32>) outs(%23 : memref<f32>) attrs =  {"adjoint of "} {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.subf %arg15, %53 : f32
        %79 = math.exp %78 : f32
        %80 = arith.mulf %arg16, %79 : f32
        %81 = arith.negf %80 : f32
        %82 = arith.addf %81, %arg17 : f32
        linalg.yield %82 : f32
      }
      %59 = memref.load %23[] : memref<f32>
      linalg.copy(%1, %24) : memref<10xf32>, memref<10xf32> 
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%16, %22 : memref<10xf32>, memref<10xf32>) outs(%24 : memref<10xf32>) attrs =  {"adjoint of ", "gradient space for "} {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.subf %arg15, %53 : f32
        %79 = math.exp %78 : f32
        %80 = arith.mulf %arg16, %79 : f32
        %81 = arith.addf %80, %arg17 : f32
        linalg.yield %81 : f32
      }
      %60 = scf.for %arg15 = %c1 to %c10 step %c1 iter_args(%arg16 = %59) -> (f32) {
        %78 = arith.subi %c10, %arg15 : index
        %79 = memref.load %17[%78] {"cached "} : memref<9xi1>
        %80 = select %79, %arg16, %cst_0 : f32
        %81 = select %79, %cst_0, %arg16 : f32
        %82 = memref.load %24[%78] : memref<10xf32>
        %83 = arith.addf %82, %80 : f32
        memref.store %83, %24[%78] : memref<10xf32>
        scf.yield %81 : f32
      }
      %61 = memref.load %24[%c0] : memref<10xf32>
      %62 = arith.addf %61, %60 : f32
      memref.store %62, %24[%c0] : memref<10xf32>
      linalg.copy(%arg13, %32) : memref<10xf32>, memref<10xf32> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%24 : memref<10xf32>) outs(%32 : memref<10xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 : f32
        linalg.yield %78 : f32
      }
      %63 = memref.cast %24 : memref<10xf32> to memref<?xf32, #map1>
      %64 = memref.cast %14 : memref<512xf32> to memref<?xf32, #map1>
      %65 = memref.cast %25 : memref<10x512xf32> to memref<?x?xf32, #map0>
      call @souter(%63, %64, %65) : (memref<?xf32, #map1>, memref<?xf32, #map1>, memref<?x?xf32, #map0>) -> ()
      linalg.copy(%arg9, %33) : memref<10x512xf32>, memref<10x512xf32> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%25 : memref<10x512xf32>) outs(%33 : memref<10x512xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 : f32
        linalg.yield %78 : f32
      }
      linalg.copy(%2, %26) : memref<512xf32>, memref<512xf32> 
      %66 = memref.cast %24 : memref<10xf32> to memref<?xf32, #map1>
      %67 = memref.cast %arg6 : memref<10x512xf32> to memref<?x?xf32, #map0>
      %68 = memref.cast %26 : memref<512xf32> to memref<?xf32, #map1>
      call @svecmat(%66, %67, %68) : (memref<?xf32, #map1>, memref<?x?xf32, #map0>, memref<?xf32, #map1>) -> ()
      linalg.copy(%2, %27) : memref<512xf32>, memref<512xf32> 
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%13, %26 : memref<512xf32>, memref<512xf32>) outs(%27 : memref<512xf32>) attrs =  {"adjoint of %h1"} {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.cmpf ogt, %arg15, %cst_0 : f32
        %79 = select %78, %arg16, %cst_0 : f32
        %80 = arith.addf %79, %arg17 : f32
        linalg.yield %80 : f32
      }
      linalg.copy(%arg10, %34) : memref<512xf32>, memref<512xf32> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%27 : memref<512xf32>) outs(%34 : memref<512xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 : f32
        linalg.yield %78 : f32
      }
      %69 = memref.cast %27 : memref<512xf32> to memref<?xf32, #map1>
      %70 = memref.cast %11 : memref<512xf32> to memref<?xf32, #map1>
      %71 = memref.cast %28 : memref<512x512xf32> to memref<?x?xf32, #map0>
      call @souter(%69, %70, %71) : (memref<?xf32, #map1>, memref<?xf32, #map1>, memref<?x?xf32, #map0>) -> ()
      linalg.copy(%arg12, %35) : memref<512x512xf32>, memref<512x512xf32> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%28 : memref<512x512xf32>) outs(%35 : memref<512x512xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 : f32
        linalg.yield %78 : f32
      }
      linalg.copy(%2, %29) : memref<512xf32>, memref<512xf32> 
      %72 = memref.cast %27 : memref<512xf32> to memref<?xf32, #map1>
      %73 = memref.cast %arg4 : memref<512x512xf32> to memref<?x?xf32, #map0>
      %74 = memref.cast %29 : memref<512xf32> to memref<?xf32, #map1>
      call @svecmat(%72, %73, %74) : (memref<?xf32, #map1>, memref<?x?xf32, #map0>, memref<?xf32, #map1>) -> ()
      linalg.copy(%2, %30) : memref<512xf32>, memref<512xf32> 
      linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%10, %29 : memref<512xf32>, memref<512xf32>) outs(%30 : memref<512xf32>) attrs =  {"adjoint of %h0"} {
      ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):  // no predecessors
        %78 = arith.cmpf ogt, %arg15, %cst_0 : f32
        %79 = select %78, %arg16, %cst_0 : f32
        %80 = arith.addf %79, %arg17 : f32
        linalg.yield %80 : f32
      }
      linalg.copy(%arg14, %36) : memref<512xf32>, memref<512xf32> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%30 : memref<512xf32>) outs(%36 : memref<512xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 : f32
        linalg.yield %78 : f32
      }
      %75 = memref.cast %30 : memref<512xf32> to memref<?xf32, #map1>
      %76 = memref.cast %40 : memref<784xf32, #map1> to memref<?xf32, #map1>
      %77 = memref.cast %31 : memref<512x784xf32> to memref<?x?xf32, #map0>
      call @souter(%75, %76, %77) : (memref<?xf32, #map1>, memref<?xf32, #map1>, memref<?x?xf32, #map0>) -> ()
      linalg.copy(%arg11, %37) : memref<512x784xf32>, memref<512x784xf32> 
      linalg.generic {doc = "Add in place", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%31 : memref<512x784xf32>) outs(%37 : memref<512x784xf32>) {
      ^bb0(%arg15: f32, %arg16: f32):  // no predecessors
        %78 = arith.addf %arg15, %arg16 : f32
        linalg.yield %78 : f32
      }
      scf.yield %33, %34, %37, %35, %32, %36 : memref<10x512xf32>, memref<512xf32>, memref<512x784xf32>, memref<512x512xf32>, memref<10xf32>, memref<512xf32>
    }
    memref.dealloc %31 : memref<512x784xf32>
    memref.dealloc %30 : memref<512xf32>
    memref.dealloc %29 : memref<512xf32>
    memref.dealloc %28 : memref<512x512xf32>
    memref.dealloc %27 : memref<512xf32>
    memref.dealloc %26 : memref<512xf32>
    memref.dealloc %25 : memref<10x512xf32>
    memref.dealloc %14 : memref<512xf32>
    memref.dealloc %13 : memref<512xf32>
    memref.dealloc %12 : memref<512xf32>
    memref.dealloc %11 : memref<512xf32>
    memref.dealloc %10 : memref<512xf32>
    memref.dealloc %9 : memref<512xf32>
    memref.dealloc %8 : memref<512xf32>
    memref.dealloc %6 : memref<512x512xf32>
    memref.dealloc %5 : memref<512x784xf32>
    memref.dealloc %4 : memref<512xf32>
    memref.dealloc %3 : memref<10x512xf32>
    return %38#2, %38#5, %38#3, %38#1, %38#0, %38#4 : memref<512x784xf32>, memref<512xf32>, memref<512x512xf32>, memref<512xf32>, memref<10x512xf32>, memref<10xf32>
  }
  func @lagrad_mlp_batched(%arg0: memref<784x64xf32>, %arg1: memref<64xi32>, %arg2: memref<512x784xf32>, %arg3: memref<512xf32>, %arg4: memref<512x512xf32>, %arg5: memref<512xf32>, %arg6: memref<10x512xf32>, %arg7: memref<10xf32>) -> (memref<512x784xf32>, memref<512xf32>, memref<512x512xf32>, memref<512xf32>, memref<10x512xf32>, memref<10xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c63 = arith.constant 63 : index
    %cst_0 = arith.constant 1.562500e-02 : f32
    %0 = memref.get_global @__constant_64xf32 : memref<64xf32>
    %1 = memref.get_global @__constant_10x64xf32 : memref<10x64xf32>
    %2 = memref.get_global @__constant_512xf32 : memref<512xf32>
    %3 = memref.get_global @__constant_512x64xf32 : memref<512x64xf32>
    %4 = memref.get_global @__constant_10xf32 : memref<10xf32>
    %5 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%3, %5) : memref<512x64xf32>, memref<512x64xf32> 
    %6 = memref.cast %arg2 : memref<512x784xf32> to memref<?x?xf32, #map0>
    %7 = memref.cast %arg0 : memref<784x64xf32> to memref<?x?xf32, #map0>
    %8 = memref.cast %5 : memref<512x64xf32> to memref<?x?xf32, #map0>
    call @smatmul(%6, %7, %8) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    %9 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel"]} ins(%5, %arg3 : memref<512x64xf32>, memref<512xf32>) outs(%9 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %5 : memref<512x64xf32>
    %10 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "ReLU 2D", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%9 : memref<512x64xf32>) outs(%10 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.cmpf ogt, %arg8, %cst : f32
      %69 = select %68, %arg8, %cst : f32
      linalg.yield %69 : f32
    }
    %11 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%3, %11) : memref<512x64xf32>, memref<512x64xf32> 
    %12 = memref.cast %arg4 : memref<512x512xf32> to memref<?x?xf32, #map0>
    %13 = memref.cast %10 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %14 = memref.cast %11 : memref<512x64xf32> to memref<?x?xf32, #map0>
    call @smatmul(%12, %13, %14) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    // linalg.matmul ins(%arg4, %10 : memref<512x512xf32>, memref<512x64xf32>) outs(%11 : memref<512x64xf32>)

    // %s = memref.subview %11[0, 0] [1, 64] [1, 1] : memref<512x64xf32> to memref<64xf32>
    // %U = memref.cast %s : memref<64xf32> to memref<*xf32>
    // call @print_memref_f32(%U) : (memref<*xf32>) -> ()

    %15 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel"]} ins(%11, %arg5 : memref<512x64xf32>, memref<512xf32>) outs(%15 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %11 : memref<512x64xf32>
    %16 = memref.alloc() : memref<512x64xf32>
    linalg.generic {doc = "ReLU 2D", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%15 : memref<512x64xf32>) outs(%16 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.cmpf ogt, %arg8, %cst : f32
      %69 = select %68, %arg8, %cst : f32
      linalg.yield %69 : f32
    }
    %17 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%1, %17) : memref<10x64xf32>, memref<10x64xf32> 
    %18 = memref.cast %arg6 : memref<10x512xf32> to memref<?x?xf32, #map0>
    %19 = memref.cast %16 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %20 = memref.cast %17 : memref<10x64xf32> to memref<?x?xf32, #map0>
    call @smatmul(%18, %19, %20) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    %21 = memref.alloc() : memref<10x64xf32>
    linalg.generic {doc = "Broadcasted add", indexing_maps = [#map2, #map6, #map2], iterator_types = ["parallel", "parallel"]} ins(%17, %arg7 : memref<10x64xf32>, memref<10xf32>) outs(%21 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %17 : memref<10x64xf32>
    %22 = memref.subview %21[0, 0] [1, 64] [1, 1] : memref<10x64xf32> to memref<64xf32, #map1>
    %23 = memref.alloca() : memref<64xf32>
    linalg.copy(%22, %23) : memref<64xf32, #map1>, memref<64xf32> 
    %24 = memref.alloca() : memref<64x9xi1>
    scf.for %arg8 = %c1 to %c10 step %c1 {
      scf.for %arg9 = %c0 to %c64 step %c1 {
        %68 = memref.load %21[%arg8, %arg9] : memref<10x64xf32>
        %69 = memref.load %23[%arg9] : memref<64xf32>
        %70 = arith.cmpf ogt, %68, %69 : f32
        memref.store %70, %24[%arg9, %arg8] {lagrad_cache} : memref<64x9xi1>
        %71 = select %70, %68, %69 : f32
        memref.store %71, %23[%arg9] : memref<64xf32>
      }
    }
    %25 = memref.alloc() : memref<10x64xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%21, %23 : memref<10x64xf32>, memref<64xf32>) outs(%25 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.subf %arg8, %arg9 : f32
      %69 = math.exp %68 : f32
      linalg.yield %69 : f32
    }
    %26 = memref.alloca() : memref<64xf32>
    linalg.copy(%0, %26) : memref<64xf32>, memref<64xf32> 
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel"]} ins(%25 : memref<10x64xf32>) outs(%26 : memref<64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    %27 = memref.alloca() : memref<64xf32>
    linalg.fill(%cst, %27) : f32, memref<64xf32> 
    %28 = memref.alloc() : memref<10x64xf32>
    linalg.fill(%cst, %28) : f32, memref<10x64xf32> 
    scf.for %arg8 = %c0 to %c64 step %c1 {
      %68 = arith.subi %c63, %arg8 : index
      %69 = memref.load %arg1[%68] : memref<64xi32>
      %70 = arith.index_cast %69 {"cloned "} : i32 to index
      %71 = memref.load %25[%70, %68] : memref<10x64xf32>
      %72 = memref.load %26[%68] : memref<64xf32>
      %73 = arith.divf %71, %72 {"cloned "} : f32
      %74 = arith.negf %cst_0 : f32
      %75 = arith.divf %74, %73 : f32
      %76 = arith.divf %75, %72 : f32
      %77 = arith.mulf %75, %71 : f32
      %78 = arith.negf %77 : f32
      %79 = arith.mulf %72, %72 : f32
      %80 = arith.divf %78, %79 : f32
      %81 = memref.load %27[%68] : memref<64xf32>
      %82 = arith.addf %81, %80 : f32
      memref.store %82, %27[%68] : memref<64xf32>
      %83 = memref.load %28[%70, %68] : memref<10x64xf32>
      %84 = arith.addf %83, %76 : f32
      memref.store %84, %28[%70, %68] : memref<10x64xf32>
    }
    memref.dealloc %25 : memref<10x64xf32>
    %29 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%1, %29) : memref<10x64xf32>, memref<10x64xf32> 
    linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%27 : memref<64xf32>) outs(%29 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    %30 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%28, %30) : memref<10x64xf32>, memref<10x64xf32> 
    memref.dealloc %28 : memref<10x64xf32>
    linalg.generic {doc = "Add in place", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%29 : memref<10x64xf32>) outs(%30 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %29 : memref<10x64xf32>
    %31 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%1, %31) : memref<10x64xf32>, memref<10x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map3, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%21, %23, %30 : memref<10x64xf32>, memref<64xf32>, memref<10x64xf32>) outs(%31 : memref<10x64xf32>) attrs =  {"adjoint of ", "gradient space for "} {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32, %arg11: f32):  // no predecessors
      %68 = arith.subf %arg8, %arg9 : f32
      %69 = math.exp %68 : f32
      %70 = arith.mulf %arg10, %69 : f32
      %71 = arith.addf %70, %arg11 : f32
      linalg.yield %71 : f32
    }
    %32 = memref.alloca() : memref<64xf32>
    linalg.copy(%0, %32) : memref<64xf32>, memref<64xf32> 
    linalg.generic {indexing_maps = [#map2, #map3, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%21, %23, %30 : memref<10x64xf32>, memref<64xf32>, memref<10x64xf32>) outs(%32 : memref<64xf32>) attrs =  {"adjoint of "} {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32, %arg11: f32):  // no predecessors
      %68 = arith.subf %arg8, %arg9 : f32
      %69 = math.exp %68 : f32
      %70 = arith.mulf %arg10, %69 : f32
      %71 = arith.negf %70 : f32
      %72 = arith.addf %71, %arg11 : f32
      linalg.yield %72 : f32
    }
    memref.dealloc %30 : memref<10x64xf32>
    memref.dealloc %21 : memref<10x64xf32>
    scf.for %arg8 = %c1 to %c10 step %c1 {
      %68 = arith.subi %c10, %arg8 : index
      scf.for %arg9 = %c0 to %c64 step %c1 {
        %69 = arith.subi %c63, %arg9 : index
        %70 = memref.load %24[%69, %68] {"cached "} : memref<64x9xi1>
        %71 = memref.load %32[%69] : memref<64xf32>
        memref.store %cst, %32[%69] : memref<64xf32>
        %72 = select %70, %71, %cst : f32
        %73 = select %70, %cst, %71 : f32
        %74 = memref.load %32[%69] : memref<64xf32>
        %75 = arith.addf %74, %73 : f32
        memref.store %75, %32[%69] : memref<64xf32>
        %76 = memref.load %31[%68, %69] : memref<10x64xf32>
        %77 = arith.addf %76, %72 : f32
        memref.store %77, %31[%68, %69] : memref<10x64xf32>
      }
    }
    %33 = memref.subview %31[0, 0] [1, 64] [1, 1] : memref<10x64xf32> to memref<64xf32, #map1>
    %34 = memref.cast %33 : memref<64xf32, #map1> to memref<64xf32>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%34, %32 : memref<64xf32>, memref<64xf32>) outs(%33 : memref<64xf32, #map1>) {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    %35 = memref.alloc() : memref<10x64xf32>
    linalg.copy(%1, %35) : memref<10x64xf32>, memref<10x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%31 : memref<10x64xf32>) outs(%35 : memref<10x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    %36 = memref.alloc() : memref<10xf32>
    linalg.copy(%4, %36) : memref<10xf32>, memref<10xf32> 
    linalg.generic {indexing_maps = [#map2, #map6], iterator_types = ["parallel", "parallel"]} ins(%31 : memref<10x64xf32>) outs(%36 : memref<10xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %31 : memref<10x64xf32>
    %37 = memref.alloc() : memref<10x512xf32>
    linalg.fill(%cst, %37) : f32, memref<10x512xf32> 
    %38 = memref.alloc() : memref<10x512xf32>
    linalg.copy(%37, %38) : memref<10x512xf32>, memref<10x512xf32> 
    memref.dealloc %37 : memref<10x512xf32>
    %39 = memref.cast %35 : memref<10x64xf32> to memref<?x?xf32, #map0>
    %40 = memref.cast %16 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %41 = memref.cast %38 : memref<10x512xf32> to memref<?x?xf32, #map0>
    call @smatmul_grad_first(%39, %40, %41) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    memref.dealloc %16 : memref<512x64xf32>
    %42 = memref.alloc() : memref<512x64xf32>
    linalg.fill(%cst, %42) : f32, memref<512x64xf32> 
    %43 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%42, %43) : memref<512x64xf32>, memref<512x64xf32> 
    memref.dealloc %42 : memref<512x64xf32>
    %44 = memref.cast %arg6 : memref<10x512xf32> to memref<?x?xf32, #map0>
    %45 = memref.cast %35 : memref<10x64xf32> to memref<?x?xf32, #map0>
    %46 = memref.cast %43 : memref<512x64xf32> to memref<?x?xf32, #map0>
    call @smatmul_grad_second(%44, %45, %46) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()

    // linalg.generic
    //   {
    //     doc = "Manual matmul second arg grad",
    //     indexing_maps = [
    //       affine_map<(d0, d1, d2) -> (d1, d0)>,
    //       affine_map<(d0, d1, d2) -> (d1, d2)>,
    //       affine_map<(d0, d1, d2) -> (d0, d2)>
    //     ],
    //     iterator_types = ["parallel", "reduction", "parallel"]
    //   }
    //   ins(%arg6, %35 : memref<10x512xf32>, memref<10x64xf32>)
    //   outs(%43 : memref<512x64xf32>) {
    // ^bb0(%marg0: f32, %marg1: f32, %marg2: f32):
    //   %m0 = arith.mulf %marg0, %marg1 : f32
    //   %m1 = arith.addf %m0, %marg2 : f32
    //   linalg.yield %m1 : f32
    // }
    memref.dealloc %35 : memref<10x64xf32>
    %47 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%3, %47) : memref<512x64xf32>, memref<512x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%15, %43 : memref<512x64xf32>, memref<512x64xf32>) outs(%47 : memref<512x64xf32>) attrs =  {"adjoint of %h1"} {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.cmpf ogt, %arg8, %cst : f32
      %69 = select %68, %arg9, %cst : f32
      %70 = arith.addf %69, %arg10 : f32
      linalg.yield %70 : f32
    }
    memref.dealloc %43 : memref<512x64xf32>
    memref.dealloc %15 : memref<512x64xf32>
    %48 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%3, %48) : memref<512x64xf32>, memref<512x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%47 : memref<512x64xf32>) outs(%48 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }

    // %s = memref.subview %47[0, 0] [1, 64] [1, 1] : memref<512x64xf32> to memref<64xf32>
    // %U = memref.cast %s : memref<64xf32> to memref<*xf32>
    // call @print_memref_f32(%U) : (memref<*xf32>) -> ()

    %49 = memref.alloc() : memref<512xf32>
    // %2 is zero
    linalg.copy(%2, %49) : memref<512xf32>, memref<512xf32> 
    linalg.generic {indexing_maps = [#map2, #map6], iterator_types = ["parallel", "parallel"]} ins(%47 : memref<512x64xf32>) outs(%49 : memref<512xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %47 : memref<512x64xf32>
    %50 = memref.alloc() : memref<512x512xf32>
    linalg.fill(%cst, %50) : f32, memref<512x512xf32> 
    %51 = memref.alloc() : memref<512x512xf32>
    linalg.copy(%50, %51) : memref<512x512xf32>, memref<512x512xf32> 
    memref.dealloc %50 : memref<512x512xf32>
    %52 = memref.cast %48 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %53 = memref.cast %10 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %54 = memref.cast %51 : memref<512x512xf32> to memref<?x?xf32, #map0>
    call @smatmul_grad_first(%52, %53, %54) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    memref.dealloc %10 : memref<512x64xf32>
    %55 = memref.alloc() : memref<512x64xf32>
    linalg.fill(%cst, %55) : f32, memref<512x64xf32> 
    %56 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%55, %56) : memref<512x64xf32>, memref<512x64xf32> 
    memref.dealloc %55 : memref<512x64xf32>
    %57 = memref.cast %arg4 : memref<512x512xf32> to memref<?x?xf32, #map0>
    %58 = memref.cast %48 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %59 = memref.cast %56 : memref<512x64xf32> to memref<?x?xf32, #map0>
    call @smatmul_grad_second(%57, %58, %59) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    memref.dealloc %48 : memref<512x64xf32>
    %60 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%3, %60) : memref<512x64xf32>, memref<512x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%9, %56 : memref<512x64xf32>, memref<512x64xf32>) outs(%60 : memref<512x64xf32>) attrs =  {"adjoint of %h0"} {
    ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
      %68 = arith.cmpf ogt, %arg8, %cst : f32
      %69 = select %68, %arg9, %cst : f32
      %70 = arith.addf %69, %arg10 : f32
      linalg.yield %70 : f32
    }
    memref.dealloc %56 : memref<512x64xf32>
    memref.dealloc %9 : memref<512x64xf32>
    %61 = memref.alloc() : memref<512x64xf32>
    linalg.copy(%3, %61) : memref<512x64xf32>, memref<512x64xf32> 
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%60 : memref<512x64xf32>) outs(%61 : memref<512x64xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    %62 = memref.alloc() : memref<512xf32>
    linalg.copy(%2, %62) : memref<512xf32>, memref<512xf32> 
    linalg.generic {indexing_maps = [#map2, #map6], iterator_types = ["parallel", "parallel"]} ins(%60 : memref<512x64xf32>) outs(%62 : memref<512xf32>) {
    ^bb0(%arg8: f32, %arg9: f32):  // no predecessors
      %68 = arith.addf %arg8, %arg9 : f32
      linalg.yield %68 : f32
    }
    memref.dealloc %60 : memref<512x64xf32>
    %63 = memref.alloc() : memref<512x784xf32>
    linalg.fill(%cst, %63) : f32, memref<512x784xf32> 
    %64 = memref.alloc() : memref<512x784xf32>
    linalg.copy(%63, %64) : memref<512x784xf32>, memref<512x784xf32> 
    memref.dealloc %63 : memref<512x784xf32>
    %65 = memref.cast %61 : memref<512x64xf32> to memref<?x?xf32, #map0>
    %66 = memref.cast %arg0 : memref<784x64xf32> to memref<?x?xf32, #map0>
    %67 = memref.cast %64 : memref<512x784xf32> to memref<?x?xf32, #map0>
    call @smatmul_grad_first(%65, %66, %67) : (memref<?x?xf32, #map0>, memref<?x?xf32, #map0>, memref<?x?xf32, #map0>) -> ()
    memref.dealloc %61 : memref<512x64xf32>
    return %64, %62, %51, %49, %38, %36 : memref<512x784xf32>, memref<512xf32>, memref<512x512xf32>, memref<512xf32>, memref<10x512xf32>, memref<10xf32>
  }
}

