#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map9 = affine_map<(d0) -> (d0)>
#map10 = affine_map<(d0, d1) -> (d1)>
#map11 = affine_map<(d0) -> ()>
#map12 = affine_map<(d0, d1, d2) -> (d0)>
#map13 = affine_map<() -> ()>
module  {
  func private @print_memref_f64(tensor<*xf64>) attributes {llvm.emit_c_interface}
  func @__grad_lagrad_gmm_objective_tri(%arg0: tensor<200xf64>, %arg1: tensor<200x128xf64>, %arg2: tensor<200x128xf64>, %arg3: tensor<200x8128xf64>, %arg4: tensor<1000x128xf64>, %arg5: f64, %arg6: i64) -> (tensor<200xf64>, tensor<200x128xf64>, tensor<200x128xf64>, tensor<200x8128xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %c128 = arith.constant 128 : index
    %c200 = arith.constant 200 : index
    %cst_0 = arith.constant 5.000000e-01 : f64
    %cst_1 = arith.constant dense<-1.000000e+09> : tensor<1000xf64>
    %cst_2 = arith.constant dense<1.000000e+03> : tensor<f64>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<f64>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<1000xf64>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<200xf64>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<1000x200xf64>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<1000x200x128xf64>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<200x128xf64>
    %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<200x128xf64>) outs(%arg2 : tensor<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %61 = math.exp %arg7 : f64
      linalg.yield %61 : f64
    } -> tensor<200x128xf64>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<200x128xf64>) outs(%cst_6 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %61 = arith.addf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<200xf64>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1 : tensor<1000x128xf64>, tensor<200x128xf64>) outs(%cst_8 : tensor<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.subf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<1000x200x128xf64>
    %3 = memref.alloc() : memref<1000x200x128xf64>
    linalg.fill(%cst, %3) : f64, memref<1000x200x128xf64> 
    %4 = memref.tensor_load %3 : memref<1000x200x128xf64>
    %5:2 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %4, %arg9 = %c0) -> (tensor<1000x200x128xf64>, index) {
      %61:2 = scf.for %arg10 = %c0 to %c200 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<1000x200x128xf64>, index) {
        %62:2 = scf.for %arg13 = %c0 to %c128 step %c1 iter_args(%arg14 = %arg11, %arg15 = %c0) -> (tensor<1000x200x128xf64>, index) {
          %63:2 = scf.for %arg16 = %c0 to %arg13 step %c1 iter_args(%arg17 = %arg14, %arg18 = %arg15) -> (tensor<1000x200x128xf64>, index) {
            %64 = tensor.extract %arg3[%arg10, %arg18] : tensor<200x8128xf64>
            %65 = tensor.extract %2[%arg7, %arg10, %arg16] : tensor<1000x200x128xf64>
            %66 = tensor.extract %arg17[%arg7, %arg10, %arg13] : tensor<1000x200x128xf64>
            %67 = arith.mulf %64, %65 : f64
            %68 = arith.addf %67, %66 : f64
            %69 = tensor.insert %68 into %arg17[%arg7, %arg10, %arg13] : tensor<1000x200x128xf64>
            %70 = arith.addi %arg18, %c1 : index
            scf.yield %69, %70 : tensor<1000x200x128xf64>, index
          }
          scf.yield %63#0, %63#1 : tensor<1000x200x128xf64>, index
        }
        scf.yield %62#0, %62#1 : tensor<1000x200x128xf64>, index
      }
      scf.yield %61#0, %61#1 : tensor<1000x200x128xf64>, index
    }
    %6 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %2, %5#0 : tensor<200x128xf64>, tensor<1000x200x128xf64>, tensor<1000x200x128xf64>) outs(%cst_8 : tensor<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %61 = arith.mulf %arg7, %arg8 : f64
      %62 = arith.addf %61, %arg9 : f64
      %63 = arith.mulf %62, %62 : f64
      linalg.yield %63 : f64
    } -> tensor<1000x200x128xf64>
    %7 = linalg.generic {indexing_maps = [#map4, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6 : tensor<1000x200x128xf64>) outs(%cst_7 : tensor<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %61 = arith.addf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<1000x200xf64>
    %8 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%arg0, %1 : tensor<200xf64>, tensor<200xf64>) outs(%arg0 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<200xf64>
    %9 = linalg.generic {indexing_maps = [#map10, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%8, %7 : tensor<200xf64>, tensor<1000x200xf64>) outs(%cst_7 : tensor<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.mulf %arg8, %cst_0 : f64
      %62 = arith.subf %arg7, %61 : f64
      linalg.yield %62 : f64
    } -> tensor<1000x200xf64>
    %10 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%9 : tensor<1000x200xf64>) outs(%cst_1 : tensor<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %61 = arith.cmpf ogt, %arg7, %arg8 : f64
      %62 = select %61, %arg7, %arg8 : f64
      linalg.yield %62 : f64
    } -> tensor<1000xf64>
    %11 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%9, %10 : tensor<1000x200xf64>, tensor<1000xf64>) outs(%cst_5 : tensor<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.subf %arg7, %arg8 : f64
      %62 = math.exp %61 : f64
      %63 = arith.addf %62, %arg9 : f64
      linalg.yield %63 : f64
    } -> tensor<1000xf64>
    %12 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel"]} ins(%11 : tensor<1000xf64>) outs(%11 : tensor<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %61 = math.log %arg7 : f64
      linalg.yield %61 : f64
    } -> tensor<1000xf64>
    %13 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%12, %10 : tensor<1000xf64>, tensor<1000xf64>) outs(%12 : tensor<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<1000xf64>
    %14 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<200x128xf64>) outs(%cst_6 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %61 = arith.mulf %arg7, %arg7 : f64
      %62 = arith.addf %61, %arg8 : f64
      linalg.yield %62 : f64
    } -> tensor<200xf64>
    %15 = memref.alloc() : memref<200xf64>
    linalg.fill(%cst, %15) : f64, memref<200xf64> 
    %16 = memref.tensor_load %15 : memref<200xf64>
    %17:2 = scf.for %arg7 = %c0 to %c200 step %c1 iter_args(%arg8 = %16, %arg9 = %c0) -> (tensor<200xf64>, index) {
      %61:2 = scf.for %arg10 = %c0 to %c128 step %c1 iter_args(%arg11 = %arg8, %arg12 = %c0) -> (tensor<200xf64>, index) {
        %62:2 = scf.for %arg13 = %c0 to %arg10 step %c1 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<200xf64>, index) {
          %63 = tensor.extract %arg3[%arg7, %arg15] : tensor<200x8128xf64>
          %64 = tensor.extract %arg14[%arg7] : tensor<200xf64>
          %65 = arith.mulf %63, %63 : f64
          %66 = arith.addf %65, %64 : f64
          %67 = tensor.insert %66 into %arg14[%arg7] : tensor<200xf64>
          %68 = arith.addi %arg15, %c1 : index
          scf.yield %67, %68 : tensor<200xf64>, index
        }
        scf.yield %62#0, %62#1 : tensor<200xf64>, index
      }
      scf.yield %61#0, %61#1 : tensor<200xf64>, index
    }
    %18 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%14, %17#0 : tensor<200xf64>, tensor<200xf64>) outs(%14 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<200xf64>
    %19 = linalg.generic {indexing_maps = [#map9, #map11], iterator_types = ["reduction"]} ins(%arg0 : tensor<200xf64>) outs(%cst_4 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %61 = math.exp %arg7 : f64
      %62 = arith.addf %61, %arg8 : f64
      linalg.yield %62 : f64
    } -> tensor<f64>
    %20 = linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%cst_3 : tensor<f64>) outs(%cst_3 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %61 = arith.negf %arg7 : f64
      linalg.yield %61 : f64
    } -> tensor<f64>
    %21 = linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%20, %cst_2 : tensor<f64>, tensor<f64>) outs(%20 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<f64>
    %22 = linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = []} ins(%21, %19 : tensor<f64>, tensor<f64>) outs(%21 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.divf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<f64>
    %23 = linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%arg0, %22 : tensor<200xf64>, tensor<f64>) outs(%cst_6 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = math.exp %arg7 : f64
      %62 = arith.mulf %arg8, %61 : f64
      %63 = arith.addf %62, %arg9 : f64
      linalg.yield %63 : f64
    } -> tensor<200xf64>
    %24 = linalg.generic {indexing_maps = [#map13, #map13], iterator_types = []} ins(%cst_3 : tensor<f64>) outs(%cst_3 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %61 = arith.negf %arg7 : f64
      linalg.yield %61 : f64
    } -> tensor<f64>
    %25 = linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%1, %24 : tensor<200xf64>, tensor<f64>) outs(%cst_6 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.sitofp %arg6 : i64 to f64
      %62 = arith.mulf %arg8, %61 : f64
      %63 = arith.addf %62, %arg9 : f64
      linalg.yield %63 : f64
    } -> tensor<200xf64>
    %26 = linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%18, %cst_3 : tensor<200xf64>, tensor<f64>) outs(%cst_6 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.mulf %cst_0, %arg5 : f64
      %62 = arith.mulf %61, %arg5 : f64
      %63 = arith.mulf %arg8, %62 : f64
      %64 = arith.addf %63, %arg9 : f64
      linalg.yield %64 : f64
    } -> tensor<200xf64>
    %27 = memref.alloc() : memref<200x8128xf64>
    linalg.fill(%cst, %27) : f64, memref<200x8128xf64> 
    %28 = memref.tensor_load %27 : memref<200x8128xf64>
    %29:2 = scf.for %arg7 = %c0 to %c200 step %c1 iter_args(%arg8 = %28, %arg9 = %c0) -> (tensor<200x8128xf64>, index) {
      %61:2 = scf.for %arg10 = %c0 to %c128 step %c1 iter_args(%arg11 = %arg8, %arg12 = %c0) -> (tensor<200x8128xf64>, index) {
        %62:2 = scf.for %arg13 = %c0 to %arg10 step %c1 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<200x8128xf64>, index) {
          %63 = tensor.extract %arg3[%arg7, %arg15] : tensor<200x8128xf64>
          %64 = tensor.extract %26[%arg7] : tensor<200xf64>
          %65 = tensor.extract %arg14[%arg7, %arg15] : tensor<200x8128xf64>
          %66 = arith.mulf %64, %63 : f64
          %67 = arith.mulf %64, %63 : f64
          %68 = arith.addf %66, %67 : f64
          %69 = arith.addf %68, %65 : f64
          %70 = tensor.insert %69 into %arg14[%arg7, %arg15] : tensor<200x8128xf64>
          %71 = arith.addi %arg15, %c1 : index
          scf.yield %70, %71 : tensor<200x8128xf64>, index
        }
        scf.yield %62#0, %62#1 : tensor<200x8128xf64>, index
      }
      scf.yield %61#0, %61#1 : tensor<200x8128xf64>, index
    }
    %30 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%0, %26 : tensor<200x128xf64>, tensor<200xf64>) outs(%cst_9 : tensor<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.mulf %arg8, %arg7 : f64
      %62 = arith.mulf %arg8, %arg7 : f64
      %63 = arith.addf %61, %62 : f64
      %64 = arith.addf %63, %arg9 : f64
      linalg.yield %64 : f64
    } -> tensor<200x128xf64>
    %31 = linalg.generic {indexing_maps = [#map9, #map11, #map9], iterator_types = ["parallel"]} ins(%13, %cst_3 : tensor<1000xf64>, tensor<f64>) outs(%cst_5 : tensor<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg8, %arg9 : f64
      linalg.yield %61 : f64
    } -> tensor<1000xf64>
    %32 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%31, %11 : tensor<1000xf64>, tensor<1000xf64>) outs(%31 : tensor<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.divf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<1000xf64>
    %33 = linalg.generic {indexing_maps = [#map0, #map1, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%9, %10, %32 : tensor<1000x200xf64>, tensor<1000xf64>, tensor<1000xf64>) outs(%cst_7 : tensor<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %61 = arith.subf %arg7, %arg8 : f64
      %62 = math.exp %61 : f64
      %63 = arith.mulf %arg9, %62 : f64
      %64 = arith.addf %63, %arg10 : f64
      linalg.yield %64 : f64
    } -> tensor<1000x200xf64>
    %34 = linalg.generic {indexing_maps = [#map0, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %10, %32 : tensor<1000x200xf64>, tensor<1000xf64>, tensor<1000xf64>) outs(%cst_5 : tensor<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %61 = arith.subf %arg7, %arg8 : f64
      %62 = math.exp %61 : f64
      %63 = arith.mulf %arg9, %62 : f64
      %64 = arith.negf %63 : f64
      %65 = arith.addf %64, %arg10 : f64
      linalg.yield %65 : f64
    } -> tensor<1000xf64>
    %35 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%31, %34 : tensor<1000xf64>, tensor<1000xf64>) outs(%31 : tensor<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<1000xf64>
    %36 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%9, %35 : tensor<1000x200xf64>, tensor<1000xf64>) outs(%cst_7 : tensor<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.cmpf ogt, %arg7, %arg9 : f64
      %62 = select %61, %arg8, %cst : f64
      %63 = arith.addf %62, %arg9 : f64
      linalg.yield %63 : f64
    } -> tensor<1000x200xf64>
    %37 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%33, %36 : tensor<1000x200xf64>, tensor<1000x200xf64>) outs(%33 : tensor<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<1000x200xf64>
    %38 = linalg.generic {indexing_maps = [#map10, #map0, #map0, #map10], iterator_types = ["parallel", "parallel"]} ins(%8, %7, %37 : tensor<200xf64>, tensor<1000x200xf64>, tensor<1000x200xf64>) outs(%cst_6 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %61 = arith.addf %arg9, %arg10 : f64
      linalg.yield %61 : f64
    } -> tensor<200xf64>
    %39 = linalg.generic {indexing_maps = [#map10, #map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%8, %7, %37 : tensor<200xf64>, tensor<1000x200xf64>, tensor<1000x200xf64>) outs(%cst_7 : tensor<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %61 = arith.negf %arg9 : f64
      %62 = arith.mulf %61, %cst_0 : f64
      %63 = arith.addf %62, %arg10 : f64
      linalg.yield %63 : f64
    } -> tensor<1000x200xf64>
    %40 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%23, %38 : tensor<200xf64>, tensor<200xf64>) outs(%23 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<200xf64>
    %41 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel"]} ins(%25, %38 : tensor<200xf64>, tensor<200xf64>) outs(%25 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<200xf64>
    %42 = linalg.generic {indexing_maps = [#map4, #map8, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %39 : tensor<1000x200x128xf64>, tensor<1000x200xf64>) outs(%cst_8 : tensor<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg8, %arg9 : f64
      linalg.yield %61 : f64
    } -> tensor<1000x200x128xf64>
    %43 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %2, %5#0, %42 : tensor<200x128xf64>, tensor<1000x200x128xf64>, tensor<1000x200x128xf64>, tensor<1000x200x128xf64>) outs(%cst_9 : tensor<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
      %61 = arith.mulf %arg7, %arg8 : f64
      %62 = arith.addf %61, %arg9 : f64
      %63 = arith.mulf %arg10, %62 : f64
      %64 = arith.mulf %arg10, %62 : f64
      %65 = arith.addf %63, %64 : f64
      %66 = arith.mulf %65, %arg8 : f64
      %67 = arith.addf %66, %arg11 : f64
      linalg.yield %67 : f64
    } -> tensor<200x128xf64>
    %44 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%30, %43 : tensor<200x128xf64>, tensor<200x128xf64>) outs(%30 : tensor<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<200x128xf64>
    %45 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %2, %5#0, %42 : tensor<200x128xf64>, tensor<1000x200x128xf64>, tensor<1000x200x128xf64>, tensor<1000x200x128xf64>) outs(%cst_8 : tensor<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
      %61 = arith.mulf %arg7, %arg8 : f64
      %62 = arith.addf %61, %arg9 : f64
      %63 = arith.mulf %arg10, %62 : f64
      %64 = arith.mulf %arg10, %62 : f64
      %65 = arith.addf %63, %64 : f64
      %66 = arith.mulf %65, %arg7 : f64
      %67 = arith.addf %66, %arg11 : f64
      linalg.yield %67 : f64
    } -> tensor<1000x200x128xf64>
    %46 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %2, %5#0, %42 : tensor<200x128xf64>, tensor<1000x200x128xf64>, tensor<1000x200x128xf64>, tensor<1000x200x128xf64>) outs(%cst_8 : tensor<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
      %61 = arith.mulf %arg7, %arg8 : f64
      %62 = arith.addf %61, %arg9 : f64
      %63 = arith.mulf %arg10, %62 : f64
      %64 = arith.mulf %arg10, %62 : f64
      %65 = arith.addf %63, %64 : f64
      %66 = arith.addf %65, %arg11 : f64
      linalg.yield %66 : f64
    } -> tensor<1000x200x128xf64>
    %47 = memref.alloc() : memref<200x8128xf64>
    linalg.fill(%cst, %47) : f64, memref<200x8128xf64> 
    %48 = memref.tensor_load %47 : memref<200x8128xf64>
    %49:2 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %48, %arg9 = %c0) -> (tensor<200x8128xf64>, index) {
      %61:2 = scf.for %arg10 = %c0 to %c200 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<200x8128xf64>, index) {
        %62:2 = scf.for %arg13 = %c0 to %c128 step %c1 iter_args(%arg14 = %arg11, %arg15 = %c0) -> (tensor<200x8128xf64>, index) {
          %63:2 = scf.for %arg16 = %c0 to %arg13 step %c1 iter_args(%arg17 = %arg14, %arg18 = %arg15) -> (tensor<200x8128xf64>, index) {
            %64 = tensor.extract %2[%arg7, %arg10, %arg16] : tensor<1000x200x128xf64>
            %65 = tensor.extract %46[%arg7, %arg10, %arg13] : tensor<1000x200x128xf64>
            %66 = tensor.extract %arg17[%arg10, %arg18] : tensor<200x8128xf64>
            %67 = arith.mulf %65, %64 : f64
            %68 = arith.addf %67, %66 : f64
            %69 = tensor.insert %68 into %arg17[%arg10, %arg18] : tensor<200x8128xf64>
            %70 = arith.addi %arg18, %c1 : index
            scf.yield %69, %70 : tensor<200x8128xf64>, index
          }
          scf.yield %63#0, %63#1 : tensor<200x8128xf64>, index
        }
        scf.yield %62#0, %62#1 : tensor<200x8128xf64>, index
      }
      scf.yield %61#0, %61#1 : tensor<200x8128xf64>, index
    }
    %50 = memref.alloc() : memref<200x8128xf64>
    linalg.fill(%cst, %50) : f64, memref<200x8128xf64> 
    %51 = memref.tensor_load %50 : memref<200x8128xf64>
    %52:2 = scf.for %arg7 = %c0 to %c200 step %c1 iter_args(%arg8 = %51, %arg9 = %c0) -> (tensor<200x8128xf64>, index) {
      %61:2 = scf.for %arg10 = %c0 to %c128 step %c1 iter_args(%arg11 = %arg8, %arg12 = %c0) -> (tensor<200x8128xf64>, index) {
        %62:2 = scf.for %arg13 = %c0 to %arg10 step %c1 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<200x8128xf64>, index) {
          %63 = tensor.extract %29#0[%arg7, %arg15] : tensor<200x8128xf64>
          %64 = tensor.extract %49#0[%arg7, %arg15] : tensor<200x8128xf64>
          %65 = arith.addf %63, %64 : f64
          %66 = tensor.insert %65 into %arg14[%arg7, %arg15] : tensor<200x8128xf64>
          %67 = arith.addi %arg15, %c1 : index
          scf.yield %66, %67 : tensor<200x8128xf64>, index
        }
        scf.yield %62#0, %62#1 : tensor<200x8128xf64>, index
      }
      scf.yield %61#0, %61#1 : tensor<200x8128xf64>, index
    }
    %53 = memref.alloc() : memref<1000x200x128xf64>
    linalg.fill(%cst, %53) : f64, memref<1000x200x128xf64> 
    %54 = memref.tensor_load %53 : memref<1000x200x128xf64>
    %55:2 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %54, %arg9 = %c0) -> (tensor<1000x200x128xf64>, index) {
      %61:2 = scf.for %arg10 = %c0 to %c200 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<1000x200x128xf64>, index) {
        %62:2 = scf.for %arg13 = %c0 to %c128 step %c1 iter_args(%arg14 = %arg11, %arg15 = %c0) -> (tensor<1000x200x128xf64>, index) {
          %63:2 = scf.for %arg16 = %c0 to %arg13 step %c1 iter_args(%arg17 = %arg14, %arg18 = %arg15) -> (tensor<1000x200x128xf64>, index) {
            %64 = tensor.extract %arg3[%arg10, %arg18] : tensor<200x8128xf64>
            %65 = tensor.extract %46[%arg7, %arg10, %arg13] : tensor<1000x200x128xf64>
            %66 = tensor.extract %arg17[%arg7, %arg10, %arg16] : tensor<1000x200x128xf64>
            %67 = arith.mulf %65, %64 : f64
            %68 = arith.addf %67, %66 : f64
            %69 = tensor.insert %68 into %arg17[%arg7, %arg10, %arg16] : tensor<1000x200x128xf64>
            %70 = arith.addi %arg18, %c1 : index
            scf.yield %69, %70 : tensor<1000x200x128xf64>, index
          }
          scf.yield %63#0, %63#1 : tensor<1000x200x128xf64>, index
        }
        scf.yield %62#0, %62#1 : tensor<1000x200x128xf64>, index
      }
      scf.yield %61#0, %61#1 : tensor<1000x200x128xf64>, index
    }
    %56 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %55#0 : tensor<1000x200x128xf64>, tensor<1000x200x128xf64>) outs(%45 : tensor<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<1000x200x128xf64>
    %57 = linalg.generic {indexing_maps = [#map2, #map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1, %56 : tensor<1000x128xf64>, tensor<200x128xf64>, tensor<1000x200x128xf64>) outs(%cst_9 : tensor<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %61 = arith.negf %arg9 : f64
      %62 = arith.addf %61, %arg10 : f64
      linalg.yield %62 : f64
    } -> tensor<200x128xf64>
    %58 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %41 : tensor<200x128xf64>, tensor<200xf64>) outs(%cst_9 : tensor<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg8, %arg9 : f64
      linalg.yield %61 : f64
    } -> tensor<200x128xf64>
    %59 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%44, %0 : tensor<200x128xf64>, tensor<200x128xf64>) outs(%44 : tensor<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<200x128xf64>
    %60 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%58, %59 : tensor<200x128xf64>, tensor<200x128xf64>) outs(%58 : tensor<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %61 = arith.addf %arg7, %arg8 : f64
      linalg.yield %61 : f64
    } -> tensor<200x128xf64>
    return %40, %57, %60, %52#0 : tensor<200xf64>, tensor<200x128xf64>, tensor<200x128xf64>, tensor<200x8128xf64>
  }

  func @lagrad_gmm_compressed(%arg0: tensor<200xf64>, %arg1: tensor<200x128xf64>, %arg2: tensor<200x128xf64>, %arg3: tensor<200x8128xf64>, %arg4: tensor<1000x128xf64>, %arg5: f64, %arg6: i64) -> (tensor<200xf64>, tensor<200x128xf64>, tensor<200x128xf64>, tensor<200x8128xf64>) {
    %0:4 = call @__grad_lagrad_gmm_objective_tri(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (tensor<200xf64>, tensor<200x128xf64>, tensor<200x128xf64>, tensor<200x8128xf64>, tensor<1000x128xf64>, f64, i64) -> (tensor<200xf64>, tensor<200x128xf64>, tensor<200x128xf64>, tensor<200x8128xf64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<200xf64>, tensor<200x128xf64>, tensor<200x128xf64>, tensor<200x8128xf64>
  }
}

