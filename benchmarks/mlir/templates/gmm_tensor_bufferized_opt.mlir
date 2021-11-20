#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#slice = affine_map<(d0)[s0] -> (d0 + s0)>
#map2 = affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>
#map3 = affine_map<(d0, d1)[s0] -> (d0 * 8128 + s0 + d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
module  {
  memref.global "private" constant @__constant_200xf64 : memref<200xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  func @cQtimesx(%arg0: memref<128xf64, #slice>, %arg1: memref<8128xf64, #slice>, %arg2: memref<128xf64>, %arg3: memref<128xf64>) {
    %c255 = arith.constant 255 : index
    %c2 = arith.constant 2 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %arg2 : memref<128xf64, #slice>, memref<128xf64>) outs(%arg3 : memref<128xf64>) {
    ^bb0(%arg4: f64, %arg5: f64, %arg6: f64):  // no predecessors
      %2 = arith.mulf %arg4, %arg5 : f64
      linalg.yield %2 : f64
    }
    %1 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %arg3) -> (memref<128xf64>) {
      %2 = arith.subi %c255, %arg4 : index
      %3 = arith.muli %2, %arg4 : index
      %4 = arith.divsi %3, %c2 : index
      %5 = arith.addi %arg4, %c1 : index
      %6:2 = scf.for %arg6 = %5 to %c128 step %c1 iter_args(%arg7 = %4, %arg8 = %arg5) -> (index, memref<128xf64>) {
        %7 = memref.load %arg1[%arg7] : memref<8128xf64, #slice>
        %8 = memref.load %arg2[%arg4] : memref<128xf64>
        %9 = memref.load %arg3[%arg6] : memref<128xf64>
        %10 = arith.mulf %7, %8 : f64
        %11 = arith.addf %10, %9 : f64
        memref.store %11, %arg3[%arg6] : memref<128xf64>
        %12 = arith.addi %arg7, %c1 : index
        scf.yield %12, %arg3 : index, memref<128xf64>
      }
      scf.yield %6#1 : memref<128xf64>
    }
    return
  }
  func @msqnorm(%arg0: memref<128xf64, #slice>) -> f64 {
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %1 = memref.alloc() : memref<f64>
    linalg.copy(%0, %1) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<128xf64, #slice>) outs(%1 : memref<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):  // no predecessors
      %3 = arith.mulf %arg1, %arg1 : f64
      %4 = arith.addf %3, %arg2 : f64
      linalg.yield %4 : f64
    }
    %2 = memref.load %1[] : memref<f64>
    memref.dealloc %1 : memref<f64>
    return %2 : f64
  }
  func @msqnorm_2d(%arg0: memref<8128xf64>) -> f64 {
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %1 = memref.alloc() : memref<f64>
    linalg.copy(%0, %1) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<8128xf64>) outs(%1 : memref<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):  // no predecessors
      %3 = arith.mulf %arg1, %arg1 : f64
      %4 = arith.addf %3, %arg2 : f64
      linalg.yield %4 : f64
    }
    %2 = memref.load %1[] : memref<f64>
    memref.dealloc %1 : memref<f64>
    return %2 : f64
  }
  func @mlogsumexp(%arg0: memref<200xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<f64>
    %1 = memref.load %arg0[%c0] : memref<200xf64>
    memref.store %1, %0[] : memref<f64>
    %2 = memref.alloc() : memref<f64>
    linalg.copy(%0, %2) : memref<f64>, memref<f64> 
    memref.dealloc %0 : memref<f64>
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<200xf64>) outs(%2 : memref<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):  // no predecessors
      %9 = arith.cmpf ogt, %arg1, %arg2 : f64
      %10 = select %9, %arg1, %arg2 : f64
      linalg.yield %10 : f64
    }
    %3 = memref.load %2[] : memref<f64>
    memref.dealloc %2 : memref<f64>
    %4 = memref.get_global @__constant_xf64 : memref<f64>
    %5 = memref.alloc() : memref<f64>
    linalg.copy(%4, %5) : memref<f64>, memref<f64> 
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<200xf64>) outs(%5 : memref<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):  // no predecessors
      %9 = arith.subf %arg1, %3 : f64
      %10 = math.exp %9 : f64
      %11 = arith.addf %10, %arg2 : f64
      linalg.yield %11 : f64
    }
    %6 = memref.load %5[] : memref<f64>
    memref.dealloc %5 : memref<f64>
    %7 = math.log %6 : f64
    %8 = arith.addf %7, %3 : f64
    return %8 : f64
  }
  func @mlog_wishart_prior(%arg0: f64, %arg1: i64, %arg2: memref<200xf64>, %arg3: memref<200x128xf64>, %arg4: memref<200x8128xf64>) -> f64 {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e-01 : f64
    %c200 = arith.constant 200 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg5 = %c0 to %c200 step %c1 iter_args(%arg6 = %cst) -> (f64) {
      %2 = memref.subview %arg3[%arg5, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64, #slice>
      %4 = call @msqnorm(%2) : (memref<128xf64, #slice>) -> f64
      %5 = memref.alloc() : memref<1x8128xf64>
      %6 = memref.subview %arg4[%arg5, 0] [1, 8128] [1, 1] : memref<200x8128xf64> to memref<1x8128xf64, #map3>
      linalg.copy(%6, %5) : memref<1x8128xf64, #map3>, memref<1x8128xf64> 
      %7 = memref.collapse_shape %5 [[0, 1]] : memref<1x8128xf64> into memref<8128xf64>
      %8 = call @msqnorm_2d(%7) : (memref<8128xf64>) -> f64
      memref.dealloc %5 : memref<1x8128xf64>
      %9 = arith.addf %4, %8 : f64
      %10 = arith.mulf %arg0, %arg0 : f64
      %11 = arith.mulf %10, %cst_0 : f64
      %12 = arith.mulf %11, %9 : f64
      %13 = arith.sitofp %arg1 : i64 to f64
      %14 = memref.load %arg2[%arg5] : memref<200xf64>
      %15 = arith.mulf %13, %14 : f64
      %16 = arith.subf %12, %15 : f64
      %17 = arith.addf %arg6, %16 : f64
      scf.yield %17 : f64
    }
    return %0 : f64
  }
  func private @print_memref_f64(memref<*xf64>) attributes {llvm.emit_c_interface}
  func @enzyme_gmm_opt_compressed(%arg0: memref<200xf64>, %arg1: memref<200x128xf64>, %arg2: memref<200x128xf64>, %arg3: memref<200x8128xf64>, %arg4: memref<1000x128xf64>, %arg5: f64, %arg6: i64) -> f64 {
    %cst = arith.constant 1.000000e+03 : f64
    %c200 = arith.constant 200 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst_0 = arith.constant 5.000000e-01 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %0 = memref.get_global @__constant_200xf64 : memref<200xf64>
    %1 = memref.alloc() : memref<128xf64>
    %2 = memref.alloc() : memref<200xf64>
    %xcentered = memref.alloc() : memref<128xf64>
    %3 = memref.alloc() : memref<200x128xf64> // %Qdiags
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<200x128xf64>) outs(%3 : memref<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %11 = math.exp %arg7 : f64
      linalg.yield %11 : f64
    }
    %4 = memref.alloc() : memref<200xf64> // sumQs
    linalg.copy(%0, %4) : memref<200xf64>, memref<200xf64> 
    linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<200x128xf64>) outs(%4 : memref<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %11 = arith.addf %arg7, %arg8 : f64
      linalg.yield %11 : f64
    }
    %5 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %cst_1) -> (f64) {
      scf.for %arg9 = %c0 to %c200 step %c1 {
        %14 = memref.subview %arg4[%arg7, 0] [1, 128] [1, 1] : memref<1000x128xf64> to memref<128xf64, #slice>
        %17 = memref.subview %arg1[%arg9, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64, #slice>
        linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%14, %17 : memref<128xf64, #slice>, memref<128xf64, #slice>) outs(%xcentered : memref<128xf64>) {
        ^bb0(%arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
          %32 = arith.subf %arg10, %arg11 : f64
          linalg.yield %32 : f64
        }

        %21 = memref.subview %3[%arg9, 0] [1, 128] [1, 1] : memref<200x128xf64> to memref<128xf64, #slice>
        %24 = memref.subview %arg3[%arg9, 0] [1, 8128] [1, 1] : memref<200x8128xf64> to memref<8128xf64, #slice>
        call @cQtimesx(%21, %24, %xcentered, %1) : (memref<128xf64, #slice>, memref<8128xf64, #slice>, memref<128xf64>, memref<128xf64>) -> ()

        %p0 = arith.cmpi "eq", %arg7, %c2 : index
        %p1 = arith.cmpi "eq", %arg9, %c1 : index
        %p = arith.andi %p0, %p1 : i1
        scf.if %p {
          %temp_out = memref.alloc() : memref<{{d}}xf64>
          linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%21, %xcentered : memref<128xf64, #slice>, memref<128xf64>) outs(%temp_out : memref<128xf64>) {
          ^bb0(%arg40: f64, %arg50: f64, %arg60: f64):  // no predecessors
            %2000 = arith.mulf %arg40, %arg50 : f64
            linalg.yield %2000 : f64
          }
          %U = memref.cast %temp_out : memref<{{d}}xf64> to memref<*xf64>
          call @print_memref_f64(%U) : (memref<*xf64>) -> ()
          memref.dealloc %temp_out : memref<{{d}}xf64>
        }
        // Bit of a hack to satisfy the type system when calling @msqnorm
        %subviewed = memref.subview %1[0] [128] [1] : memref<128xf64> to memref<128xf64, #slice>
        %26 = call @msqnorm(%subviewed) : (memref<128xf64, #slice>) -> f64
        %27 = arith.mulf %26, %cst_0 : f64
        %28 = memref.load %arg0[%arg9] : memref<200xf64>
        %29 = memref.load %4[%arg9] : memref<200xf64>
        %30 = arith.addf %28, %29 : f64
        %31 = arith.subf %30, %27 : f64
        memref.store %31, %2[%arg9] : memref<200xf64>
      }
      %11 = call @mlogsumexp(%2) : (memref<200xf64>) -> f64
      %12 = arith.addf %arg8, %11 : f64
      scf.yield %12 : f64
    }
    memref.dealloc %2 : memref<200xf64>
    memref.dealloc %1 : memref<128xf64>
    memref.dealloc %xcentered : memref<128xf64>
    %6 = call @mlogsumexp(%arg0) : (memref<200xf64>) -> f64
    %7 = arith.mulf %cst, %6 : f64
    %8 = call @mlog_wishart_prior(%arg5, %arg6, %4, %3, %arg3) : (f64, i64, memref<200xf64>, memref<200x128xf64>, memref<200x8128xf64>) -> f64
    memref.dealloc %4 : memref<200xf64>
    memref.dealloc %3 : memref<200x128xf64>
    %9 = arith.subf %5, %7 : f64
    %10 = arith.addf %9, %8 : f64
    return %10 : f64
  }

  // {% if method == 'DISABLED_enzyme_mlir_compressed' %}
  func @enzyme_gmm_opt_diff_compressed(
    %arg0: memref<{{k}}xf64>,
    %arg1: memref<{{k}}x{{d}}xf64>,
    %arg2: memref<{{k}}x{{d}}xf64>,
    %arg3: memref<{{k}}x{{tri_size}}xf64>,
    %arg4: memref<{{n}}x{{d}}xf64>,
    %arg5: f64,
    %arg6: i64
  ) -> (
    memref<{{k}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{d}}xf64>,
    memref<{{k}}x{{tri_size}}xf64>
  ) {
    %zero = arith.constant 0.0 : f64
    %darg0 = memref.alloc() : memref<{{k}}xf64>
    %darg1 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    %darg2 = memref.alloc() : memref<{{k}}x{{d}}xf64>
    %darg3 = memref.alloc() : memref<{{k}}x{{tri_size}}xf64>

    linalg.fill(%zero, %darg0) : f64, memref<{{k}}xf64>
    linalg.fill(%zero, %darg1) : f64, memref<{{k}}x{{d}}xf64>
    linalg.fill(%zero, %darg2) : f64, memref<{{k}}x{{d}}xf64>
    linalg.fill(%zero, %darg3) : f64, memref<{{k}}x{{tri_size}}xf64>

    %f = constant @enzyme_gmm_opt_compressed : (memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{tri_size}}xf64>, memref<{{n}}x{{d}}xf64>, f64, i64) -> f64
    %df = standalone.diff %f {const = [4]} : (
      memref<{{k}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{tri_size}}xf64>,
      memref<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> f64, (
      memref<{{k}}xf64>,
      memref<{{k}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{tri_size}}xf64>,
      memref<{{k}}x{{tri_size}}xf64>,
      memref<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> f64
    call_indirect %df(%arg0, %darg0, %arg1, %darg1, %arg2, %darg2, %arg3, %darg3, %arg4, %arg5, %arg6) : (
      memref<{{k}}xf64>,
      memref<{{k}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{d}}xf64>,
      memref<{{k}}x{{tri_size}}xf64>,
      memref<{{k}}x{{tri_size}}xf64>,
      memref<{{n}}x{{d}}xf64>,
      f64,
      i64
    ) -> f64
    return %darg0, %darg1, %darg2, %darg3 : memref<{{k}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{d}}xf64>, memref<{{k}}x{{tri_size}}xf64>
  }
  // {% endif %}

}

