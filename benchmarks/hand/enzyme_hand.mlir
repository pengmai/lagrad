#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#mapS = affine_map<(d0)[s0] -> (d0 + s0)>
#map2 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map3 = affine_map<(d0, d1) -> (d0 * 4 + d1)>
#map4 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s0 + d1 * s2 + s1)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map6 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map8 = affine_map<(d0, d1) -> (d0, d1)>
#map9 = affine_map<(d0, d1) -> (d0)>
#map10 = affine_map<(d0, d1) -> (d1)>
module  {
  memref.global "private" constant @__constant_544x4xf64 : memref<{{nverts}}x4xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_22x4x4xf64 : memref<{{nbones}}x4x4xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_544x3xf64 : memref<{{nverts}}x3xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_3xf64_0 : memref<3xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_4x4xf64 : memref<4x4xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_3x3xf64_0 : memref<3x3xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_3x3xf64 : memref<3x3xf64> = sparse<[[0, 0], [1, 1], [2, 2]], 1.000000e+00>
  memref.global "private" constant @__constant_3xf64 : memref<3xf64> = dense<1.000000e+00>
  func @eto_pose_params(%arg0: memref<{{ntheta}}xf64>) -> memref<{{nbones + 3}}x3xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %0 = memref.get_global @__constant_3xf64 : memref<3xf64>
    %1 = memref.alloc() : memref<{{nbones + 3}}x3xf64>
    linalg.fill(%cst, %1) : f64, memref<{{nbones + 3}}x3xf64> 
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %2 = scf.for %arg1 = %c0 to %c3 step %c1 iter_args(%arg2 = %1) -> (memref<{{nbones + 3}}x3xf64>) {
      %4 = memref.load %arg0[%arg1] : memref<{{ntheta}}xf64>
      %5 = arith.addi %arg1, %c3 : index
      %6 = memref.load %arg0[%5] : memref<{{ntheta}}xf64>
      memref.store %4, %arg2[%c0, %arg1] : memref<{{nbones + 3}}x3xf64>
      memref.store %cst_0, %arg2[%c1, %arg1] : memref<{{nbones + 3}}x3xf64>
      memref.store %6, %arg2[%c2, %arg1] : memref<{{nbones + 3}}x3xf64>
      scf.yield %arg2 : memref<{{nbones + 3}}x3xf64>
    }
    %c6 = arith.constant 6 : index
    %c5_1 = arith.constant 5 : index
    %3:3 = scf.for %arg1 = %c0 to %c5 step %c1 iter_args(%arg2 = %2, %arg3 = %c6, %arg4 = %c5_1) -> (memref<{{nbones + 3}}x3xf64>, index, index) {
      %4 = memref.load %arg0[%arg3] : memref<{{ntheta}}xf64>
      %5 = arith.addi %arg3, %c1 : index
      %6 = memref.load %arg0[%5] : memref<{{ntheta}}xf64>
      %7 = arith.addi %5, %c1 : index
      %8 = memref.load %arg0[%7] : memref<{{ntheta}}xf64>
      %9 = arith.addi %7, %c1 : index
      %10 = memref.load %arg0[%9] : memref<{{ntheta}}xf64>
      %11 = arith.addi %9, %c1 : index
      memref.store %4, %arg2[%arg4, %c0] : memref<{{nbones + 3}}x3xf64>
      memref.store %6, %arg2[%arg4, %c1] : memref<{{nbones + 3}}x3xf64>
      %12 = arith.addi %arg4, %c1 : index
      memref.store %8, %arg2[%12, %c0] : memref<{{nbones + 3}}x3xf64>
      %13 = arith.addi %12, %c1 : index
      memref.store %10, %arg2[%13, %c0] : memref<{{nbones + 3}}x3xf64>
      %14 = arith.addi %13, %c2 : index
      scf.yield %arg2, %11, %14 : memref<{{nbones + 3}}x3xf64>, index, index
    }
    return %3#0 : memref<{{nbones + 3}}x3xf64>
  }
  func @eangle_axis_to_rotation_matrix(%arg0: memref<3xf64>) -> memref<3x3xf64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %0 = memref.get_global @__constant_3x3xf64 : memref<3x3xf64>
    %1 = memref.get_global @__constant_xf64 : memref<f64>
    %2 = memref.alloc() : memref<f64>
    linalg.copy(%1, %2) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<3xf64>) outs(%2 : memref<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):  // no predecessors
      %8 = arith.mulf %arg1, %arg1 : f64
      %9 = arith.addf %8, %arg2 : f64
      linalg.yield %9 : f64
    }
    %3 = memref.load %2[] : memref<f64>
    %4 = math.sqrt %3 : f64
    %5 = memref.alloc() : memref<3x3xf64>
    %cst_1 = arith.constant 1.000000e-04 : f64
    %6 = arith.cmpf olt, %4, %cst_1 : f64
    %7 = scf.if %6 -> (memref<3x3xf64>) {
      scf.yield %0 : memref<3x3xf64>
    } else {
      %8 = memref.load %arg0[%c0] : memref<3xf64>
      %9 = memref.load %arg0[%c1] : memref<3xf64>
      %10 = memref.load %arg0[%c2] : memref<3xf64>
      %11 = arith.divf %8, %4 : f64
      %12 = arith.divf %9, %4 : f64
      %13 = arith.divf %10, %4 : f64
      %14 = math.sin %4 : f64
      %15 = math.cos %4 : f64
      %16 = arith.mulf %11, %11 : f64
      %17 = arith.mulf %12, %12 : f64
      %18 = arith.mulf %13, %13 : f64
      %19 = arith.mulf %11, %12 : f64
      %20 = arith.mulf %11, %13 : f64
      %21 = arith.mulf %12, %13 : f64
      %22 = arith.subf %cst_0, %15 : f64
      %23 = arith.mulf %11, %14 : f64
      %24 = arith.mulf %12, %14 : f64
      %25 = arith.mulf %13, %14 : f64
      %26 = arith.subf %cst_0, %16 : f64
      %27 = arith.mulf %26, %15 : f64
      %28 = arith.addf %16, %27 : f64
      memref.store %28, %5[%c0, %c0] : memref<3x3xf64>
      %29 = arith.mulf %19, %22 : f64
      %30 = arith.subf %29, %25 : f64
      memref.store %30, %5[%c1, %c0] : memref<3x3xf64>
      %31 = arith.mulf %20, %22 : f64
      %32 = arith.addf %31, %24 : f64
      memref.store %32, %5[%c2, %c0] : memref<3x3xf64>
      %33 = arith.mulf %19, %22 : f64
      %34 = arith.addf %33, %25 : f64
      memref.store %34, %5[%c0, %c1] : memref<3x3xf64>
      %35 = arith.subf %cst_0, %17 : f64
      %36 = arith.mulf %35, %15 : f64
      %37 = arith.addf %17, %36 : f64
      memref.store %37, %5[%c1, %c1] : memref<3x3xf64>
      %38 = arith.mulf %21, %22 : f64
      %39 = arith.subf %38, %23 : f64
      memref.store %39, %5[%c2, %c1] : memref<3x3xf64>
      %40 = arith.mulf %20, %22 : f64
      %41 = arith.subf %40, %24 : f64
      memref.store %41, %5[%c0, %c2] : memref<3x3xf64>
      %42 = arith.mulf %21, %22 : f64
      %43 = arith.addf %42, %23 : f64
      memref.store %43, %5[%c1, %c2] : memref<3x3xf64>
      %44 = arith.subf %cst_0, %18 : f64
      %45 = arith.mulf %44, %15 : f64
      %46 = arith.addf %18, %45 : f64
      memref.store %46, %5[%c2, %c2] : memref<3x3xf64>
      scf.yield %5 : memref<3x3xf64>
    }
    return %7 : memref<3x3xf64>
  }
  func @eeuler_angles_to_rotation_matrix(%arg0: memref<3xf64>) -> memref<3x3xf64> attributes {pure = true} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %0 = memref.load %arg0[%c0] : memref<3xf64>
    %1 = memref.load %arg0[%c2] : memref<3xf64>
    %2 = memref.load %arg0[%c1] : memref<3xf64>
    %3 = memref.alloc() : memref<3x3xf64>
    linalg.fill(%cst, %3) : f64, memref<3x3xf64> 
    %4 = math.cos %0 : f64
    memref.store %4, %3[%c1, %c1] : memref<3x3xf64>
    %5 = math.sin %0 : f64
    memref.store %5, %3[%c1, %c2] : memref<3x3xf64>
    %6 = arith.negf %5 : f64
    memref.store %6, %3[%c2, %c1] : memref<3x3xf64>
    memref.store %4, %3[%c2, %c2] : memref<3x3xf64>
    memref.store %cst_0, %3[%c0, %c0] : memref<3x3xf64>
    %7 = memref.alloc() : memref<3x3xf64>
    linalg.fill(%cst, %7) : f64, memref<3x3xf64> 
    %8 = math.cos %1 : f64
    memref.store %8, %7[%c0, %c0] : memref<3x3xf64>
    %9 = math.sin %1 : f64
    memref.store %9, %7[%c0, %c2] : memref<3x3xf64>
    %10 = arith.negf %9 : f64
    memref.store %10, %7[%c2, %c0] : memref<3x3xf64>
    memref.store %8, %7[%c2, %c2] : memref<3x3xf64>
    memref.store %cst_0, %7[%c1, %c1] : memref<3x3xf64>
    %11 = memref.alloc() : memref<3x3xf64>
    linalg.fill(%cst, %11) : f64, memref<3x3xf64> 
    %12 = math.cos %2 : f64
    memref.store %12, %11[%c0, %c0] : memref<3x3xf64>
    %13 = math.sin %2 : f64
    memref.store %13, %11[%c0, %c1] : memref<3x3xf64>
    %14 = arith.negf %13 : f64
    memref.store %14, %11[%c1, %c0] : memref<3x3xf64>
    memref.store %12, %11[%c1, %c1] : memref<3x3xf64>
    memref.store %cst_0, %11[%c2, %c2] : memref<3x3xf64>
    %15 = memref.get_global @__constant_3x3xf64_0 : memref<3x3xf64>
    %16 = memref.alloc() : memref<3x3xf64>
    linalg.copy(%15, %16) : memref<3x3xf64>, memref<3x3xf64> 
    linalg.matmul ins(%7, %11 : memref<3x3xf64>, memref<3x3xf64>) outs(%16 : memref<3x3xf64>)
    %17 = memref.alloc() : memref<3x3xf64>
    linalg.copy(%15, %17) : memref<3x3xf64>, memref<3x3xf64> 
    linalg.matmul ins(%3, %16 : memref<3x3xf64>, memref<3x3xf64>) outs(%17 : memref<3x3xf64>)
    return %17 : memref<3x3xf64>
  }

  func @eget_skinned_vertex_positions(%0: memref<{{nbones + 3}}x3xf64>, %arg1: memref<{{nbones}}xi32>, %arg2: memref<{{nbones}}x4x4xf64>, %arg3: memref<{{nbones}}x4x4xf64>, %arg4: memref<{{nverts}}x4xf64>, %arg5: memref<{{nverts}}x{{nbones}}xf64>, %arg6: memref<{{npts}}xi32>, %arg7: memref<{{npts}}x3xf64>, %out: memref<{{nverts}}x3xf64>) {
    %cpy0 = memref.alloc() : memref<{{nbones + 3}}x3xf64>
    linalg.copy(%0, %cpy0) : memref<{{nbones + 3}}x3xf64>, memref<{{nbones + 3}}x3xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %cb = arith.constant {{nbones}} : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %1 = memref.get_global @__constant_4x4xf64 : memref<4x4xf64>
    %2 = memref.alloc() : memref<4x4xf64>
    %3 = memref.get_global @__constant_3xf64_0 : memref<3xf64>
    linalg.fill(%cst, %2) : f64, memref<4x4xf64> 
    %4 = memref.alloc() : memref<{{nbones}}x4x4xf64>

    %5 = scf.for %arg8 = %c0 to %cb step %c1 iter_args(%arg9 = %4) -> (memref<{{nbones}}x4x4xf64>) {
      %27 = arith.addi %arg8, %c3 : index
      %28 = memref.subview %cpy0[%27, 0] [1, 3] [1, 1] : memref<{{nbones + 3}}x3xf64> to memref<3xf64, #mapS>
      %29 = memref.cast %28 : memref<3xf64, #mapS> to memref<3xf64>
      %30 = memref.alloc() : memref<3xf64>
      linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%29, %3 : memref<3xf64>, memref<3xf64>) outs(%30 : memref<3xf64>) {
      ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
        %38 = arith.addf %arg10, %arg11 : f64
        linalg.yield %38 : f64
      }
      %31 = call @eeuler_angles_to_rotation_matrix(%30) : (memref<3xf64>) -> memref<3x3xf64>
      %32 = memref.alloc() : memref<4x4xf64>
      linalg.copy(%2, %32) : memref<4x4xf64>, memref<4x4xf64> 
      %33 = memref.subview %32[0, 0] [3, 3] [1, 1] : memref<4x4xf64> to memref<3x3xf64, #map3>
      linalg.copy(%31, %33) : memref<3x3xf64>, memref<3x3xf64, #map3> 
      memref.store %cst_0, %32[%c3, %c3] : memref<4x4xf64>
      %35 = memref.subview %arg2[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
      %36 = memref.alloc() : memref<4x4xf64>
      linalg.copy(%1, %36) : memref<4x4xf64>, memref<4x4xf64> 
      linalg.matmul ins(%32, %35 : memref<4x4xf64>, memref<4x4xf64, #map4>) outs(%36 : memref<4x4xf64>)
      %37 = memref.subview %arg9[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
      linalg.copy(%36, %37) : memref<4x4xf64>, memref<4x4xf64, #map4> 
      scf.yield %arg9 : memref<{{nbones}}x4x4xf64>
    }
    %c-1_i32 = arith.constant -1 : i32
    %6 = memref.alloc() : memref<{{nbones}}x4x4xf64>
    %7 = scf.for %arg8 = %c0 to %cb step %c1 iter_args(%arg9 = %6) -> (memref<{{nbones}}x4x4xf64>) {
      %27 = memref.load %arg1[%arg8] : memref<{{nbones}}xi32>
      %28 = memref.subview %5[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
      %29 = memref.cast %28 : memref<4x4xf64, #map4> to memref<4x4xf64>
      %30 = arith.cmpi eq, %27, %c-1_i32 : i32
      %31 = scf.if %30 -> (memref<4x4xf64>) {
        scf.yield %29 : memref<4x4xf64>
      } else {
        %33 = arith.index_cast %27 : i32 to index
        %34 = memref.subview %arg9[%33, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
        %35 = memref.cast %34 : memref<4x4xf64, #map4> to memref<4x4xf64>
        %36 = memref.alloc() : memref<4x4xf64>
        linalg.copy(%1, %36) : memref<4x4xf64>, memref<4x4xf64> 
        linalg.matmul ins(%29, %35 : memref<4x4xf64>, memref<4x4xf64>) outs(%36 : memref<4x4xf64>)
        scf.yield %36 : memref<4x4xf64>
      }
      %32 = memref.subview %arg9[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
      linalg.copy(%31, %32) : memref<4x4xf64>, memref<4x4xf64, #map4> 
      scf.yield %arg9 : memref<{{nbones}}x4x4xf64>
    }
    %8 = memref.get_global @__constant_22x4x4xf64 : memref<{{nbones}}x4x4xf64>
    %9 = memref.alloc() : memref<{{nbones}}x4x4xf64>
    linalg.copy(%8, %9) : memref<{{nbones}}x4x4xf64>, memref<{{nbones}}x4x4xf64> 
    linalg.batch_matmul ins(%arg3, %7 : memref<{{nbones}}x4x4xf64>, memref<{{nbones}}x4x4xf64>) outs(%9 : memref<{{nbones}}x4x4xf64>)
    %10 = memref.alloc() : memref<{{nverts}}x3xf64>
    linalg.fill(%cst, %10) : f64, memref<{{nverts}}x3xf64>
    %11 = memref.get_global @__constant_544x4xf64 : memref<{{nverts}}x4xf64>
    %12 = scf.for %arg8 = %c0 to %cb step %c1 iter_args(%arg9 = %10) -> (memref<{{nverts}}x3xf64>) {
      %27 = memref.subview %9[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
      %28 = memref.cast %27 : memref<4x4xf64, #map4> to memref<4x4xf64>
      %29 = memref.alloc() : memref<{{nverts}}x4xf64>
      linalg.copy(%11, %29) : memref<{{nverts}}x4xf64>, memref<{{nverts}}x4xf64> 
      linalg.generic {doc = "Column-major matrix multiplication", indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%28, %arg4 : memref<4x4xf64>, memref<{{nverts}}x4xf64>) outs(%29 : memref<{{nverts}}x4xf64>) {
      ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
        %35 = arith.mulf %arg10, %arg11 : f64
        %36 = arith.addf %35, %arg12 : f64
        linalg.yield %36 : f64
      }
      %30 = memref.alloc() : memref<{{nverts}}x3xf64>
      %31 = memref.subview %29[0, 0] [544, 3] [1, 1] : memref<{{nverts}}x4xf64> to memref<{{nverts}}x3xf64, #map3>
      linalg.copy(%31, %30) : memref<{{nverts}}x3xf64, #map3>, memref<{{nverts}}x3xf64> 
      %32 = memref.subview %arg5[0, %arg8] [544, 1] [1, 1] : memref<{{nverts}}x{{nbones}}xf64> to memref<{{nverts}}xf64, #map2>
      %33 = memref.cast %32 : memref<{{nverts}}xf64, #map2> to memref<{{nverts}}xf64>
      %34 = memref.alloc() : memref<{{nverts}}x3xf64>
      linalg.copy(%arg9, %34) : memref<{{nverts}}x3xf64>, memref<{{nverts}}x3xf64> 
      linalg.generic {indexing_maps = [#map8, #map9, #map8], iterator_types = ["parallel", "parallel"]} ins(%30, %33 : memref<{{nverts}}x3xf64>, memref<{{nverts}}xf64>) outs(%34 : memref<{{nverts}}x3xf64>) {
      ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
        %35 = arith.mulf %arg10, %arg11 : f64
        %36 = arith.addf %35, %arg12 : f64
        linalg.yield %36 : f64
      }
      scf.yield %34 : memref<{{nverts}}x3xf64>
    }
    %13 = memref.subview %0[0, 0] [1, 3] [1, 1] : memref<{{nbones + 3}}x3xf64> to memref<3xf64, #map2>
    %14 = memref.cast %13 : memref<3xf64, #map2> to memref<3xf64>
    %15 = memref.subview %0[1, 0] [1, 3] [1, 1] : memref<{{nbones + 3}}x3xf64> to memref<3xf64, #map2>
    %16 = memref.cast %15 : memref<3xf64, #map2> to memref<3xf64>
    %17 = memref.subview %0[2, 0] [1, 3] [1, 1] : memref<{{nbones + 3}}x3xf64> to memref<3xf64, #map2>
    %18 = memref.cast %17 : memref<3xf64, #map2> to memref<3xf64>
    %19 = call @eangle_axis_to_rotation_matrix(%14) : (memref<3xf64>) -> memref<3x3xf64>
    %20 = memref.get_global @__constant_3x3xf64_0 : memref<3x3xf64>
    %21 = memref.alloc() : memref<3x3xf64>
    linalg.generic {indexing_maps = [#map8, #map9, #map8], iterator_types = ["parallel", "parallel"]} ins(%19, %16 : memref<3x3xf64>, memref<3xf64>) outs(%21 : memref<3x3xf64>) {
    ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %27 = arith.mulf %arg8, %arg9 : f64
      linalg.yield %27 : f64
    }
    %22 = memref.get_global @__constant_544x3xf64 : memref<{{nverts}}x3xf64>
    %23 = memref.alloc() : memref<{{nverts}}x3xf64>
    linalg.copy(%22, %23) : memref<{{nverts}}x3xf64>, memref<{{nverts}}x3xf64> 
    linalg.generic {doc = "Column-major matrix multiplication", indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%21, %12 : memref<3x3xf64>, memref<{{nverts}}x3xf64>) outs(%23 : memref<{{nverts}}x3xf64>) {
    ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %27 = arith.mulf %arg8, %arg9 : f64
      %28 = arith.addf %27, %arg10 : f64
      linalg.yield %28 : f64
    }
    linalg.generic {indexing_maps = [#map8, #map10, #map8], iterator_types = ["parallel", "parallel"]} ins(%23, %18 : memref<{{nverts}}x3xf64>, memref<3xf64>) outs(%out : memref<{{nverts}}x3xf64>) {
    ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %27 = arith.addf %arg8, %arg9 : f64
      linalg.yield %27 : f64
    }
    return
  }

  func @emlir_hand_objective(%arg0: memref<{{ntheta}}xf64>, %arg1: memref<{{nbones}}xi32>, %arg2: memref<{{nbones}}x4x4xf64>, %arg3: memref<{{nbones}}x4x4xf64>, %arg4: memref<{{nverts}}x4xf64>, %arg5: memref<{{nverts}}x{{nbones}}xf64>, %arg6: memref<{{npts}}xi32>, %arg7: memref<{{npts}}x3xf64>, %out: memref<{{npts}}x3xf64>) -> f64 {
    %0 = call @eto_pose_params(%arg0) : (memref<{{ntheta}}xf64>) -> memref<{{nbones + 3}}x3xf64>
    // %24 = memref.alloc() : memref<{{nverts}}x3xf64>
    // call @eget_skinned_vertex_positions(%0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %24) : (memref<{{nbones + 3}}x3xf64>, memref<{{nbones}}xi32>, memref<{{nbones}}x4x4xf64>, memref<{{nbones}}x4x4xf64>, memref<{{nverts}}x4xf64>, memref<{{nverts}}x{{nbones}}xf64>, memref<{{npts}}xi32>, memref<{{npts}}x3xf64>, memref<{{nverts}}x3xf64>) -> ()
    %cpy0 = memref.alloc() : memref<{{nbones + 3}}x3xf64>
    linalg.copy(%0, %cpy0) : memref<{{nbones + 3}}x3xf64>, memref<{{nbones + 3}}x3xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %cb = arith.constant {{nbones}} : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %1 = memref.get_global @__constant_4x4xf64 : memref<4x4xf64>
    %2 = memref.alloc() : memref<4x4xf64>
    %3 = memref.get_global @__constant_3xf64_0 : memref<3xf64>
    linalg.fill(%cst, %2) : f64, memref<4x4xf64> 
    %4 = memref.alloc() : memref<{{nbones}}x4x4xf64>

    %5 = scf.for %arg8 = %c0 to %cb step %c1 iter_args(%arg9 = %4) -> (memref<{{nbones}}x4x4xf64>) {
      %27 = arith.addi %arg8, %c3 : index
      %28 = memref.subview %cpy0[%27, 0] [1, 3] [1, 1] : memref<{{nbones + 3}}x3xf64> to memref<3xf64, #mapS>
      %29 = memref.cast %28 : memref<3xf64, #mapS> to memref<3xf64>
      %30 = memref.alloc() : memref<3xf64>
      linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%29, %3 : memref<3xf64>, memref<3xf64>) outs(%30 : memref<3xf64>) {
      ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
        %38 = arith.addf %arg10, %arg11 : f64
        linalg.yield %38 : f64
      }
      %31 = call @eeuler_angles_to_rotation_matrix(%30) : (memref<3xf64>) -> memref<3x3xf64>
      %32 = memref.alloc() : memref<4x4xf64>
      linalg.copy(%2, %32) : memref<4x4xf64>, memref<4x4xf64> 
      %33 = memref.subview %32[0, 0] [3, 3] [1, 1] : memref<4x4xf64> to memref<3x3xf64, #map3>
      linalg.copy(%31, %33) : memref<3x3xf64>, memref<3x3xf64, #map3> 
      memref.store %cst_0, %32[%c3, %c3] : memref<4x4xf64>
      %35 = memref.subview %arg2[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
      %36 = memref.alloc() : memref<4x4xf64>
      linalg.copy(%1, %36) : memref<4x4xf64>, memref<4x4xf64> 
      linalg.matmul ins(%32, %35 : memref<4x4xf64>, memref<4x4xf64, #map4>) outs(%36 : memref<4x4xf64>)
      %37 = memref.subview %arg9[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
      linalg.copy(%36, %37) : memref<4x4xf64>, memref<4x4xf64, #map4> 
      scf.yield %arg9 : memref<{{nbones}}x4x4xf64>
    }
    %c-1_i32 = arith.constant -1 : i32
    %6 = memref.alloc() : memref<{{nbones}}x4x4xf64>
    %7 = scf.for %arg8 = %c0 to %cb step %c1 iter_args(%arg9 = %6) -> (memref<{{nbones}}x4x4xf64>) {
      %27 = memref.load %arg1[%arg8] : memref<{{nbones}}xi32>
      %28 = memref.subview %5[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
      %29 = memref.cast %28 : memref<4x4xf64, #map4> to memref<4x4xf64>
      %30 = arith.cmpi eq, %27, %c-1_i32 : i32
      %31 = scf.if %30 -> (memref<4x4xf64>) {
        scf.yield %29 : memref<4x4xf64>
      } else {
        %33 = arith.index_cast %27 : i32 to index
        %34 = memref.subview %arg9[%33, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
        %35 = memref.cast %34 : memref<4x4xf64, #map4> to memref<4x4xf64>
        %36 = memref.alloc() : memref<4x4xf64>
        linalg.copy(%1, %36) : memref<4x4xf64>, memref<4x4xf64> 
        linalg.matmul ins(%29, %35 : memref<4x4xf64>, memref<4x4xf64>) outs(%36 : memref<4x4xf64>)
        scf.yield %36 : memref<4x4xf64>
      }
      %32 = memref.subview %arg9[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
      linalg.copy(%31, %32) : memref<4x4xf64>, memref<4x4xf64, #map4> 
      scf.yield %arg9 : memref<{{nbones}}x4x4xf64>
    }
    %8 = memref.get_global @__constant_22x4x4xf64 : memref<{{nbones}}x4x4xf64>
    %9 = memref.alloc() : memref<{{nbones}}x4x4xf64>
    linalg.copy(%8, %9) : memref<{{nbones}}x4x4xf64>, memref<{{nbones}}x4x4xf64> 
    linalg.batch_matmul ins(%arg3, %7 : memref<{{nbones}}x4x4xf64>, memref<{{nbones}}x4x4xf64>) outs(%9 : memref<{{nbones}}x4x4xf64>)
    %10 = memref.alloc() : memref<{{nverts}}x3xf64>
    linalg.fill(%cst, %10) : f64, memref<{{nverts}}x3xf64>
    %11 = memref.get_global @__constant_544x4xf64 : memref<{{nverts}}x4xf64>
    %12 = scf.for %arg8 = %c0 to %cb step %c1 iter_args(%arg9 = %10) -> (memref<{{nverts}}x3xf64>) {
      %27 = memref.subview %9[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
      %28 = memref.cast %27 : memref<4x4xf64, #map4> to memref<4x4xf64>
      %29 = memref.alloc() : memref<{{nverts}}x4xf64>
      linalg.copy(%11, %29) : memref<{{nverts}}x4xf64>, memref<{{nverts}}x4xf64> 
      linalg.generic {doc = "Column-major matrix multiplication", indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%28, %arg4 : memref<4x4xf64>, memref<{{nverts}}x4xf64>) outs(%29 : memref<{{nverts}}x4xf64>) {
      ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
        %35 = arith.mulf %arg10, %arg11 : f64
        %36 = arith.addf %35, %arg12 : f64
        linalg.yield %36 : f64
      }
      %30 = memref.alloc() : memref<{{nverts}}x3xf64>
      %31 = memref.subview %29[0, 0] [544, 3] [1, 1] : memref<{{nverts}}x4xf64> to memref<{{nverts}}x3xf64, #map3>
      linalg.copy(%31, %30) : memref<{{nverts}}x3xf64, #map3>, memref<{{nverts}}x3xf64> 
      %32 = memref.subview %arg5[0, %arg8] [544, 1] [1, 1] : memref<{{nverts}}x{{nbones}}xf64> to memref<{{nverts}}xf64, #map2>
      %33 = memref.cast %32 : memref<{{nverts}}xf64, #map2> to memref<{{nverts}}xf64>
      %34 = memref.alloc() : memref<{{nverts}}x3xf64>
      linalg.copy(%arg9, %34) : memref<{{nverts}}x3xf64>, memref<{{nverts}}x3xf64> 
      linalg.generic {indexing_maps = [#map8, #map9, #map8], iterator_types = ["parallel", "parallel"]} ins(%30, %33 : memref<{{nverts}}x3xf64>, memref<{{nverts}}xf64>) outs(%34 : memref<{{nverts}}x3xf64>) {
      ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
        %35 = arith.mulf %arg10, %arg11 : f64
        %36 = arith.addf %35, %arg12 : f64
        linalg.yield %36 : f64
      }
      scf.yield %34 : memref<{{nverts}}x3xf64>
    }
    %13 = memref.subview %0[0, 0] [1, 3] [1, 1] : memref<{{nbones + 3}}x3xf64> to memref<3xf64, #map2>
    %14 = memref.cast %13 : memref<3xf64, #map2> to memref<3xf64>
    %15 = memref.subview %0[1, 0] [1, 3] [1, 1] : memref<{{nbones + 3}}x3xf64> to memref<3xf64, #map2>
    %16 = memref.cast %15 : memref<3xf64, #map2> to memref<3xf64>
    %17 = memref.subview %0[2, 0] [1, 3] [1, 1] : memref<{{nbones + 3}}x3xf64> to memref<3xf64, #map2>
    %18 = memref.cast %17 : memref<3xf64, #map2> to memref<3xf64>
    %19 = call @eangle_axis_to_rotation_matrix(%14) : (memref<3xf64>) -> memref<3x3xf64>
    %20 = memref.get_global @__constant_3x3xf64_0 : memref<3x3xf64>
    %21 = memref.alloc() : memref<3x3xf64>
    linalg.generic {indexing_maps = [#map8, #map9, #map8], iterator_types = ["parallel", "parallel"]} ins(%19, %16 : memref<3x3xf64>, memref<3xf64>) outs(%21 : memref<3x3xf64>) {
    ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %27 = arith.mulf %arg8, %arg9 : f64
      linalg.yield %27 : f64
    }
    %22 = memref.get_global @__constant_544x3xf64 : memref<{{nverts}}x3xf64>
    %23 = memref.alloc() : memref<{{nverts}}x3xf64>
    linalg.copy(%22, %23) : memref<{{nverts}}x3xf64>, memref<{{nverts}}x3xf64> 
    linalg.generic {doc = "Column-major matrix multiplication", indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%21, %12 : memref<3x3xf64>, memref<{{nverts}}x3xf64>) outs(%23 : memref<{{nverts}}x3xf64>) {
    ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %27 = arith.mulf %arg8, %arg9 : f64
      %28 = arith.addf %27, %arg10 : f64
      linalg.yield %28 : f64
    }
    %24 = memref.alloc() : memref<{{nverts}}x3xf64>
    linalg.generic {indexing_maps = [#map8, #map10, #map8], iterator_types = ["parallel", "parallel"]} ins(%23, %18 : memref<{{nverts}}x3xf64>, memref<3xf64>) outs(%24 : memref<{{nverts}}x3xf64>) {
    ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %27 = arith.addf %arg8, %arg9 : f64
      linalg.yield %27 : f64
    }
    // %cpts = memref.dim %out, %c0 : memref<{{npts}}x3xf64>
    %25 = memref.alloc() : memref<{{npts}}x3xf64>
    %cpts = arith.constant {{npts}} : index
    // %c0 = arith.constant 0 : index
    // %c1 = arith.constant 1 : index
    // %c3 = arith.constant 3 : index
    %26 = scf.for %arg8 = %c0 to %cpts step %c1 iter_args(%arg9 = %25) -> (memref<{{npts}}x3xf64>) {
      %27 = scf.for %arg10 = %c0 to %c3 step %c1 iter_args(%arg11 = %arg9) -> (memref<{{npts}}x3xf64>) {
        %28 = memref.load %arg7[%arg8, %arg10] : memref<{{npts}}x3xf64>
        %29 = memref.load %arg6[%arg8] : memref<{{npts}}xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %24[%30, %arg10] : memref<{{nverts}}x3xf64>
        %32 = arith.subf %28, %31 : f64
        memref.store %32, %arg11[%arg8, %arg10] : memref<{{npts}}x3xf64>
        scf.yield %arg11 : memref<{{npts}}x3xf64>
      }
      scf.yield %27 : memref<{{npts}}x3xf64>
    }
    linalg.copy(%26, %out) : memref<{{npts}}x3xf64>, memref<{{npts}}x3xf64>
    %ret = arith.constant 0.0 : f64
    return %ret : f64
  }

  func @emlir_hand_objective_complicated(%arg0: memref<{{ntheta}}xf64>, %us: memref<{{npts}}x2xf64>, %arg1: memref<{{nbones}}xi32>, %arg2: memref<{{nbones}}x4x4xf64>, %arg3: memref<{{nbones}}x4x4xf64>, %arg4: memref<{{nverts}}x4xf64>, %arg5: memref<{{nverts}}x{{nbones}}xf64>, %triangles: memref<{{ntriangles}}x3xi32>, %arg6: memref<{{npts}}xi32>, %arg7: memref<{{npts}}x3xf64>, %out: memref<{{npts}}x3xf64>) -> f64 {
    %0 = call @eto_pose_params(%arg0) : (memref<{{ntheta}}xf64>) -> memref<{{nbones + 3}}x3xf64>
    %24 = memref.alloc() : memref<{{nverts}}x3xf64>
    call @eget_skinned_vertex_positions(%0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %24) : (memref<{{nbones + 3}}x3xf64>, memref<{{nbones}}xi32>, memref<{{nbones}}x4x4xf64>, memref<{{nbones}}x4x4xf64>, memref<{{nverts}}x4xf64>, memref<{{nverts}}x{{nbones}}xf64>, memref<{{npts}}xi32>, memref<{{npts}}x3xf64>, memref<{{nverts}}x3xf64>) -> ()
    // %cpy0 = memref.alloc() : memref<{{nbones + 3}}x3xf64>
    // linalg.copy(%0, %cpy0) : memref<{{nbones + 3}}x3xf64>, memref<{{nbones + 3}}x3xf64>
    // %c0 = arith.constant 0 : index
    // %c1 = arith.constant 1 : index
    // %c2 = arith.constant 2 : index
    // %c3 = arith.constant 3 : index
    // %cb = arith.constant {{nbones}} : index
    // %cst = arith.constant 0.000000e+00 : f64
    // %cst_0 = arith.constant 1.000000e+00 : f64
    // %1 = memref.get_global @__constant_4x4xf64 : memref<4x4xf64>
    // %2 = memref.alloc() : memref<4x4xf64>
    // %3 = memref.get_global @__constant_3xf64_0 : memref<3xf64>
    // linalg.fill(%cst, %2) : f64, memref<4x4xf64> 
    // %4 = memref.alloc() : memref<{{nbones}}x4x4xf64>

    // %5 = scf.for %arg8 = %c0 to %cb step %c1 iter_args(%arg9 = %4) -> (memref<{{nbones}}x4x4xf64>) {
    //   %27 = arith.addi %arg8, %c3 : index
    //   %28 = memref.subview %cpy0[%27, 0] [1, 3] [1, 1] : memref<{{nbones + 3}}x3xf64> to memref<3xf64, #mapS>
    //   %29 = memref.cast %28 : memref<3xf64, #mapS> to memref<3xf64>
    //   %30 = memref.alloc() : memref<3xf64>
    //   linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%29, %3 : memref<3xf64>, memref<3xf64>) outs(%30 : memref<3xf64>) {
    //   ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
    //     %38 = arith.addf %arg10, %arg11 : f64
    //     linalg.yield %38 : f64
    //   }
    //   %31 = call @eeuler_angles_to_rotation_matrix(%30) : (memref<3xf64>) -> memref<3x3xf64>
    //   %32 = memref.alloc() : memref<4x4xf64>
    //   linalg.copy(%2, %32) : memref<4x4xf64>, memref<4x4xf64> 
    //   %33 = memref.subview %32[0, 0] [3, 3] [1, 1] : memref<4x4xf64> to memref<3x3xf64, #map3>
    //   linalg.copy(%31, %33) : memref<3x3xf64>, memref<3x3xf64, #map3> 
    //   memref.store %cst_0, %32[%c3, %c3] : memref<4x4xf64>
    //   %35 = memref.subview %arg2[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
    //   %36 = memref.alloc() : memref<4x4xf64>
    //   linalg.copy(%1, %36) : memref<4x4xf64>, memref<4x4xf64> 
    //   linalg.matmul ins(%32, %35 : memref<4x4xf64>, memref<4x4xf64, #map4>) outs(%36 : memref<4x4xf64>)
    //   %37 = memref.subview %arg9[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
    //   linalg.copy(%36, %37) : memref<4x4xf64>, memref<4x4xf64, #map4> 
    //   scf.yield %arg9 : memref<{{nbones}}x4x4xf64>
    // }
    // %c-1_i32 = arith.constant -1 : i32
    // %6 = memref.alloc() : memref<{{nbones}}x4x4xf64>
    // %7 = scf.for %arg8 = %c0 to %cb step %c1 iter_args(%arg9 = %6) -> (memref<{{nbones}}x4x4xf64>) {
    //   %27 = memref.load %arg1[%arg8] : memref<{{nbones}}xi32>
    //   %28 = memref.subview %5[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
    //   %29 = memref.cast %28 : memref<4x4xf64, #map4> to memref<4x4xf64>
    //   %30 = arith.cmpi eq, %27, %c-1_i32 : i32
    //   %31 = scf.if %30 -> (memref<4x4xf64>) {
    //     scf.yield %29 : memref<4x4xf64>
    //   } else {
    //     %33 = arith.index_cast %27 : i32 to index
    //     %34 = memref.subview %arg9[%33, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
    //     %35 = memref.cast %34 : memref<4x4xf64, #map4> to memref<4x4xf64>
    //     %36 = memref.alloc() : memref<4x4xf64>
    //     linalg.copy(%1, %36) : memref<4x4xf64>, memref<4x4xf64> 
    //     linalg.matmul ins(%29, %35 : memref<4x4xf64>, memref<4x4xf64>) outs(%36 : memref<4x4xf64>)
    //     scf.yield %36 : memref<4x4xf64>
    //   }
    //   %32 = memref.subview %arg9[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
    //   linalg.copy(%31, %32) : memref<4x4xf64>, memref<4x4xf64, #map4> 
    //   scf.yield %arg9 : memref<{{nbones}}x4x4xf64>
    // }
    // %8 = memref.get_global @__constant_22x4x4xf64 : memref<{{nbones}}x4x4xf64>
    // %9 = memref.alloc() : memref<{{nbones}}x4x4xf64>
    // linalg.copy(%8, %9) : memref<{{nbones}}x4x4xf64>, memref<{{nbones}}x4x4xf64> 
    // linalg.batch_matmul ins(%arg3, %7 : memref<{{nbones}}x4x4xf64>, memref<{{nbones}}x4x4xf64>) outs(%9 : memref<{{nbones}}x4x4xf64>)
    // %10 = memref.alloc() : memref<{{nverts}}x3xf64>
    // linalg.fill(%cst, %10) : f64, memref<{{nverts}}x3xf64>
    // %11 = memref.get_global @__constant_544x4xf64 : memref<{{nverts}}x4xf64>
    // %12 = scf.for %arg8 = %c0 to %cb step %c1 iter_args(%arg9 = %10) -> (memref<{{nverts}}x3xf64>) {
    //   %27 = memref.subview %9[%arg8, 0, 0] [1, 4, 4] [1, 1, 1] : memref<{{nbones}}x4x4xf64> to memref<4x4xf64, #map4>
    //   %28 = memref.cast %27 : memref<4x4xf64, #map4> to memref<4x4xf64>
    //   %29 = memref.alloc() : memref<{{nverts}}x4xf64>
    //   linalg.copy(%11, %29) : memref<{{nverts}}x4xf64>, memref<{{nverts}}x4xf64> 
    //   linalg.generic {doc = "Column-major matrix multiplication", indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%28, %arg4 : memref<4x4xf64>, memref<{{nverts}}x4xf64>) outs(%29 : memref<{{nverts}}x4xf64>) {
    //   ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
    //     %35 = arith.mulf %arg10, %arg11 : f64
    //     %36 = arith.addf %35, %arg12 : f64
    //     linalg.yield %36 : f64
    //   }
    //   %30 = memref.alloc() : memref<{{nverts}}x3xf64>
    //   %31 = memref.subview %29[0, 0] [544, 3] [1, 1] : memref<{{nverts}}x4xf64> to memref<{{nverts}}x3xf64, #map3>
    //   linalg.copy(%31, %30) : memref<{{nverts}}x3xf64, #map3>, memref<{{nverts}}x3xf64> 
    //   %32 = memref.subview %arg5[0, %arg8] [544, 1] [1, 1] : memref<{{nverts}}x{{nbones}}xf64> to memref<{{nverts}}xf64, #map2>
    //   %33 = memref.cast %32 : memref<{{nverts}}xf64, #map2> to memref<{{nverts}}xf64>
    //   %34 = memref.alloc() : memref<{{nverts}}x3xf64>
    //   linalg.copy(%arg9, %34) : memref<{{nverts}}x3xf64>, memref<{{nverts}}x3xf64> 
    //   linalg.generic {indexing_maps = [#map8, #map9, #map8], iterator_types = ["parallel", "parallel"]} ins(%30, %33 : memref<{{nverts}}x3xf64>, memref<{{nverts}}xf64>) outs(%34 : memref<{{nverts}}x3xf64>) {
    //   ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
    //     %35 = arith.mulf %arg10, %arg11 : f64
    //     %36 = arith.addf %35, %arg12 : f64
    //     linalg.yield %36 : f64
    //   }
    //   scf.yield %34 : memref<{{nverts}}x3xf64>
    // }
    // %13 = memref.subview %0[0, 0] [1, 3] [1, 1] : memref<{{nbones + 3}}x3xf64> to memref<3xf64, #map2>
    // %14 = memref.cast %13 : memref<3xf64, #map2> to memref<3xf64>
    // %15 = memref.subview %0[1, 0] [1, 3] [1, 1] : memref<{{nbones + 3}}x3xf64> to memref<3xf64, #map2>
    // %16 = memref.cast %15 : memref<3xf64, #map2> to memref<3xf64>
    // %17 = memref.subview %0[2, 0] [1, 3] [1, 1] : memref<{{nbones + 3}}x3xf64> to memref<3xf64, #map2>
    // %18 = memref.cast %17 : memref<3xf64, #map2> to memref<3xf64>
    // %19 = call @eangle_axis_to_rotation_matrix(%14) : (memref<3xf64>) -> memref<3x3xf64>
    // %20 = memref.get_global @__constant_3x3xf64_0 : memref<3x3xf64>
    // %21 = memref.alloc() : memref<3x3xf64>
    // linalg.generic {indexing_maps = [#map8, #map9, #map8], iterator_types = ["parallel", "parallel"]} ins(%19, %16 : memref<3x3xf64>, memref<3xf64>) outs(%21 : memref<3x3xf64>) {
    // ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
    //   %27 = arith.mulf %arg8, %arg9 : f64
    //   linalg.yield %27 : f64
    // }
    // %22 = memref.get_global @__constant_544x3xf64 : memref<{{nverts}}x3xf64>
    // %23 = memref.alloc() : memref<{{nverts}}x3xf64>
    // linalg.copy(%22, %23) : memref<{{nverts}}x3xf64>, memref<{{nverts}}x3xf64> 
    // linalg.generic {doc = "Column-major matrix multiplication", indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%21, %12 : memref<3x3xf64>, memref<{{nverts}}x3xf64>) outs(%23 : memref<{{nverts}}x3xf64>) {
    // ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
    //   %27 = arith.mulf %arg8, %arg9 : f64
    //   %28 = arith.addf %27, %arg10 : f64
    //   linalg.yield %28 : f64
    // }
    // %24 = memref.alloc() : memref<{{nverts}}x3xf64>
    // linalg.generic {indexing_maps = [#map8, #map10, #map8], iterator_types = ["parallel", "parallel"]} ins(%23, %18 : memref<{{nverts}}x3xf64>, memref<3xf64>) outs(%24 : memref<{{nverts}}x3xf64>) {
    // ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
    //   %27 = arith.addf %arg8, %arg9 : f64
    //   linalg.yield %27 : f64
    // }
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %one = arith.constant 1.0 : f64
    %cpts = arith.constant {{npts}} : index
    scf.for %arg10 = %c0 to %cpts step %c1 {
      scf.for %arg11 = %c0 to %c3 step %c1 {
        %point = memref.load %arg7[%arg10, %arg11] : memref<{{npts}}x3xf64>
        %corrs_32 = memref.load %arg6[%arg10] : memref<{{npts}}xi32>
        %corrs = arith.index_cast %corrs_32 : i32 to index
        %u_0 = memref.load %us[%arg10, %c0] : memref<{{npts}}x2xf64>
        %u_1 = memref.load %us[%arg10, %c1] : memref<{{npts}}x2xf64>
        %usum = arith.addf %u_0, %u_1 : f64
        %onemu = arith.subf %one, %usum : f64
        %verts_0_32 = memref.load %triangles[%corrs, %c0] : memref<{{ntriangles}}x3xi32>
        %verts_1_32 = memref.load %triangles[%corrs, %c1] : memref<{{ntriangles}}x3xi32>
        %verts_2_32 = memref.load %triangles[%corrs, %c2] : memref<{{ntriangles}}x3xi32>
        %verts_0 = arith.index_cast %verts_0_32 : i32 to index
        %verts_1 = arith.index_cast %verts_1_32 : i32 to index
        %verts_2 = arith.index_cast %verts_2_32 : i32 to index
        %vp0 = memref.load %24[%verts_0, %arg11] : memref<{{nverts}}x3xf64>
        %vp1 = memref.load %24[%verts_1, %arg11] : memref<{{nverts}}x3xf64>
        %vp2 = memref.load %24[%verts_2, %arg11] : memref<{{nverts}}x3xf64>

        %hand_point_coord0 = arith.mulf %u_0, %vp0 : f64
        %hand_point_coord1 = arith.mulf %u_1, %vp1 : f64
        %hand_point_coord2 = arith.mulf %onemu, %vp2 : f64
        %hand_point_coord3 = arith.addf %hand_point_coord0, %hand_point_coord1 : f64
        %hand_point_coord4 = arith.addf %hand_point_coord3, %hand_point_coord2 : f64

        %final_val = arith.subf %point, %hand_point_coord4 : f64
        memref.store %final_val, %out[%arg10, %arg11] : memref<{{npts}}x3xf64>
      }
    }
    memref.dealloc %24 : memref<{{nverts}}x3xf64>
    %ret = arith.constant 0.0 : f64
    return %ret : f64
  }

  func @enzyme_hand_objective(%arg0: memref<{{ntheta}}xf64>, %arg1: memref<{{nbones}}xi32>, %arg2: memref<{{nbones}}x4x4xf64>, %arg3: memref<{{nbones}}x4x4xf64>, %arg4: memref<{{nverts}}x4xf64>, %arg5: memref<{{nverts}}x{{nbones}}xf64>, %arg6: memref<{{npts}}xi32>, %arg7: memref<{{npts}}x3xf64>, %g: memref<{{npts}}x3xf64>) -> memref<{{ntheta}}xf64> {
    %dtheta = memref.alloc() : memref<{{ntheta}}xf64>
    // %dout = memref.alloc() : memref<{{npts}}x3xf64>
    %c0 = arith.constant 0 : index
    // %cpts = memref.dim %g, %c0 : memref<?x3xf64>
    %out = memref.alloc() : memref<{{npts}}x3xf64>
    %zero = arith.constant 0.0 : f64
    linalg.fill(%zero, %dtheta) : f64, memref<{{ntheta}}xf64>
    linalg.fill(%zero, %out) : f64, memref<{{npts}}x3xf64>
    // %one = arith.constant 1.0 : f64
    // linalg.fill(%one, %dout) : f64, memref<{{npts}}x3xf64>
    %f = constant @emlir_hand_objective : (
      memref<{{ntheta}}xf64>,
      memref<{{nbones}}xi32>,
      memref<{{nbones}}x4x4xf64>,
      memref<{{nbones}}x4x4xf64>,
      memref<{{nverts}}x4xf64>,
      memref<{{nverts}}x{{nbones}}xf64>,
      memref<{{npts}}xi32>,
      memref<{{npts}}x3xf64>,
      memref<{{npts}}x3xf64>
    ) -> f64
    %df = lagrad.diff %f {const = [1, 2, 3, 4, 5, 6, 7]} : (
      memref<{{ntheta}}xf64>,
      memref<{{nbones}}xi32>,
      memref<{{nbones}}x4x4xf64>,
      memref<{{nbones}}x4x4xf64>,
      memref<{{nverts}}x4xf64>,
      memref<{{nverts}}x{{nbones}}xf64>,
      memref<{{npts}}xi32>,
      memref<{{npts}}x3xf64>,
      memref<{{npts}}x3xf64>
    ) -> f64, (
      memref<{{ntheta}}xf64>,
      memref<{{ntheta}}xf64>,
      memref<{{nbones}}xi32>,
      memref<{{nbones}}x4x4xf64>,
      memref<{{nbones}}x4x4xf64>,
      memref<{{nverts}}x4xf64>,
      memref<{{nverts}}x{{nbones}}xf64>,
      memref<{{npts}}xi32>,
      memref<{{npts}}x3xf64>,
      memref<{{npts}}x3xf64>,
      memref<{{npts}}x3xf64>
    ) -> f64
    call_indirect %df(%arg0, %dtheta, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %out, %g) : (
      memref<{{ntheta}}xf64>,
      memref<{{ntheta}}xf64>,
      memref<{{nbones}}xi32>,
      memref<{{nbones}}x4x4xf64>,
      memref<{{nbones}}x4x4xf64>,
      memref<{{nverts}}x4xf64>,
      memref<{{nverts}}x{{nbones}}xf64>,
      memref<{{npts}}xi32>,
      memref<{{npts}}x3xf64>,
      memref<{{npts}}x3xf64>,
      memref<{{npts}}x3xf64>
    ) -> f64
    return %dtheta : memref<{{ntheta}}xf64>
  }

  func @enzyme_hand_objective_complicated(%arg0: memref<26xf64>, %arg1: memref<{{npts}}x2xf64>, %arg2: memref<22xi32>, %arg3: memref<22x4x4xf64>, %arg4: memref<22x4x4xf64>, %arg5: memref<{{nverts}}x4xf64>, %arg6: memref<{{nverts}}x22xf64>, %arg7: memref<1084x3xi32>, %arg8: memref<{{npts}}xi32>, %arg9: memref<{{npts}}x3xf64>, %dout: memref<{{npts}}x3xf64>) -> (memref<{{ntheta}}xf64>, memref<{{npts}}x2xf64>) {
    %dtheta = memref.alloc() : memref<26xf64>
    %dus = memref.alloc() : memref<{{npts}}x2xf64>
    %out = memref.alloc() : memref<{{npts}}x3xf64>
    %zero = arith.constant 0.0 : f64
    linalg.fill(%zero, %dtheta) : f64, memref<26xf64>
    linalg.fill(%zero, %dus) : f64, memref<{{npts}}x2xf64>
    linalg.fill(%zero, %out) : f64, memref<{{npts}}x3xf64>

    %f = constant @emlir_hand_objective_complicated : (
      memref<26xf64>,
      memref<{{npts}}x2xf64>,
      memref<22xi32>,
      memref<22x4x4xf64>,
      memref<22x4x4xf64>,
      memref<{{nverts}}x4xf64>,
      memref<{{nverts}}x22xf64>,
      memref<1084x3xi32>,
      memref<{{npts}}xi32>,
      memref<{{npts}}x3xf64>,
      memref<{{npts}}x3xf64>
    ) -> f64
    %df = lagrad.diff %f {const = [2, 3, 4, 5, 6, 7, 8, 9]} : (
      memref<26xf64>,
      memref<{{npts}}x2xf64>,
      memref<22xi32>,
      memref<22x4x4xf64>,
      memref<22x4x4xf64>,
      memref<{{nverts}}x4xf64>,
      memref<{{nverts}}x22xf64>,
      memref<1084x3xi32>,
      memref<{{npts}}xi32>,
      memref<{{npts}}x3xf64>,
      memref<{{npts}}x3xf64>
    ) -> f64, (
      memref<26xf64>,
      memref<26xf64>,
      memref<{{npts}}x2xf64>,
      memref<{{npts}}x2xf64>,
      memref<22xi32>,
      memref<22x4x4xf64>,
      memref<22x4x4xf64>,
      memref<{{nverts}}x4xf64>,
      memref<{{nverts}}x22xf64>,
      memref<1084x3xi32>,
      memref<{{npts}}xi32>,
      memref<{{npts}}x3xf64>,
      memref<{{npts}}x3xf64>,
      memref<{{npts}}x3xf64>
    ) -> f64

    call_indirect %df(%arg0, %dtheta, %arg1, %dus, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %out, %dout) : (
      memref<26xf64>,
      memref<26xf64>,
      memref<{{npts}}x2xf64>,
      memref<{{npts}}x2xf64>,
      memref<22xi32>,
      memref<22x4x4xf64>,
      memref<22x4x4xf64>,
      memref<{{nverts}}x4xf64>,
      memref<{{nverts}}x22xf64>,
      memref<1084x3xi32>,
      memref<{{npts}}xi32>,
      memref<{{npts}}x3xf64>,
      memref<{{npts}}x3xf64>,
      memref<{{npts}}x3xf64>
    ) -> f64
    return %dtheta, %dus : memref<26xf64>, memref<{{npts}}x2xf64>
  }
}
