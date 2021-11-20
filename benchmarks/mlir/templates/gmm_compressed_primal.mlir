#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0) -> (d0)>
#map7 = affine_map<(d0, d1) -> (d1)>
#map8 = affine_map<(d0) -> ()>
#map9 = affine_map<() -> ()>
module  {
  func private @print_memref_f64(tensor<*xf64>) attributes {llvm.emit_c_interface}
  func @lagrad_gmm_objective_compressed(%arg0: tensor<200xf64>, %arg1: tensor<200x128xf64>, %arg2: tensor<200x128xf64>, %arg3: tensor<200x8128xf64>, %arg4: tensor<1000x128xf64>, %arg5: f64, %arg6: i64) -> tensor<f64> {
    %cst = arith.constant dense<1.000000e+03> : tensor<f64>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_1 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c200 = arith.constant 200 : index
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<200xf64>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1000xf64>
    %cst_4 = arith.constant dense<-1.000000e+09> : tensor<1000xf64>
    %cst_5 = arith.constant 5.000000e-01 : f64
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<1000x200xf64>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<1000x200x128xf64>
    %c1000 = arith.constant 1000 : index
    %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<200x128xf64>) outs(%arg2 : tensor<200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %28 = math.exp %arg7 : f64
      linalg.yield %28 : f64
    } -> tensor<200x128xf64>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<200x128xf64>) outs(%cst_2 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %28 = arith.addf %arg7, %arg8 : f64
      linalg.yield %28 : f64
    } -> tensor<200xf64>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %arg1 : tensor<1000x128xf64>, tensor<200x128xf64>) outs(%cst_7 : tensor<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %28 = arith.subf %arg7, %arg8 : f64
      linalg.yield %28 : f64
    } -> tensor<1000x200x128xf64>
    %3 = memref.alloc() : memref<1000x200x128xf64>
    linalg.fill(%cst_1, %3) : f64, memref<1000x200x128xf64> 
    %4 = memref.tensor_load %3 : memref<1000x200x128xf64>
    %5:2 = scf.for %arg7 = %c0 to %c1000 step %c1 iter_args(%arg8 = %4, %arg9 = %c0) -> (tensor<1000x200x128xf64>, index) {
      %28:2 = scf.for %arg10 = %c0 to %c200 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<1000x200x128xf64>, index) {
        %29:2 = scf.for %arg13 = %c0 to %c128 step %c1 iter_args(%arg14 = %arg11, %arg15 = %c0) -> (tensor<1000x200x128xf64>, index) {
          %30:2 = scf.for %arg16 = %c0 to %arg13 step %c1 iter_args(%arg17 = %arg14, %arg18 = %arg15) -> (tensor<1000x200x128xf64>, index) {
            %31 = tensor.extract %arg3[%arg10, %arg18] : tensor<200x8128xf64>
            %32 = tensor.extract %2[%arg7, %arg10, %arg16] : tensor<1000x200x128xf64>
            %33 = tensor.extract %arg17[%arg7, %arg10, %arg13] : tensor<1000x200x128xf64>
            %34 = arith.mulf %31, %32 : f64
            %35 = arith.addf %34, %33 : f64
            %36 = tensor.insert %35 into %arg17[%arg7, %arg10, %arg13] : tensor<1000x200x128xf64>
            %37 = arith.addi %arg18, %c1 : index
            scf.yield %36, %37 : tensor<1000x200x128xf64>, index
          }
          scf.yield %30#0, %30#1 : tensor<1000x200x128xf64>, index
        }
        scf.yield %29#0, %29#1 : tensor<1000x200x128xf64>, index
      }
      scf.yield %28#0, %28#1 : tensor<1000x200x128xf64>, index
    }
    %6 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %2, %5#0 : tensor<200x128xf64>, tensor<1000x200x128xf64>, tensor<1000x200x128xf64>) outs(%cst_7 : tensor<1000x200x128xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
      %28 = arith.mulf %arg7, %arg8 : f64
      %29 = arith.addf %28, %arg9 : f64
      %30 = arith.mulf %29, %29 : f64
      linalg.yield %30 : f64
    } -> tensor<1000x200x128xf64>
    %7 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6 : tensor<1000x200x128xf64>) outs(%cst_6 : tensor<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %28 = arith.addf %arg7, %arg8 : f64
      linalg.yield %28 : f64
    } -> tensor<1000x200xf64>
    %8 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel"]} ins(%arg0, %1 : tensor<200xf64>, tensor<200xf64>) outs(%arg0 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %28 = arith.addf %arg7, %arg8 : f64
      linalg.yield %28 : f64
    } -> tensor<200xf64>
    %9 = linalg.generic {indexing_maps = [#map7, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%8, %7 : tensor<200xf64>, tensor<1000x200xf64>) outs(%cst_6 : tensor<1000x200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %28 = arith.mulf %arg8, %cst_5 : f64
      %29 = arith.subf %arg7, %28 : f64
      linalg.yield %29 : f64
    } -> tensor<1000x200xf64>
    %10 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%9 : tensor<1000x200xf64>) outs(%cst_4 : tensor<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %28 = arith.cmpf ogt, %arg7, %arg8 : f64
      %29 = select %28, %arg7, %arg8 : f64
      linalg.yield %29 : f64
    } -> tensor<1000xf64>
    %11 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%9, %10 : tensor<1000x200xf64>, tensor<1000xf64>) outs(%cst_3 : tensor<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %28 = arith.subf %arg7, %arg8 : f64
      %29 = math.exp %28 : f64
      %30 = arith.addf %29, %arg9 : f64
      linalg.yield %30 : f64
    } -> tensor<1000xf64>
    %12 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel"]} ins(%11 : tensor<1000xf64>) outs(%11 : tensor<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %28 = math.log %arg7 : f64
      linalg.yield %28 : f64
    } -> tensor<1000xf64>
    %13 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel"]} ins(%12, %10 : tensor<1000xf64>, tensor<1000xf64>) outs(%12 : tensor<1000xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %28 = arith.addf %arg7, %arg8 : f64
      linalg.yield %28 : f64
    } -> tensor<1000xf64>
    %14 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["reduction"]} ins(%13 : tensor<1000xf64>) outs(%cst_0 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %28 = arith.addf %arg7, %arg8 : f64
      linalg.yield %28 : f64
    } -> tensor<f64>
    %15 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<200x128xf64>) outs(%cst_2 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %28 = arith.mulf %arg7, %arg7 : f64
      %29 = arith.addf %28, %arg8 : f64
      linalg.yield %29 : f64
    } -> tensor<200xf64>
    %16 = memref.alloc() : memref<200xf64>
    linalg.fill(%cst_1, %16) : f64, memref<200xf64> 
    %17 = memref.tensor_load %16 : memref<200xf64>
    %18:2 = scf.for %arg7 = %c0 to %c200 step %c1 iter_args(%arg8 = %17, %arg9 = %c0) -> (tensor<200xf64>, index) {
      %28:2 = scf.for %arg10 = %c0 to %c128 step %c1 iter_args(%arg11 = %arg8, %arg12 = %c0) -> (tensor<200xf64>, index) {
        %29:2 = scf.for %arg13 = %c0 to %arg10 step %c1 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<200xf64>, index) {
          %30 = tensor.extract %arg3[%arg7, %arg15] : tensor<200x8128xf64>
          %31 = tensor.extract %arg14[%arg7] : tensor<200xf64>
          %32 = arith.mulf %30, %30 : f64
          %33 = arith.addf %32, %31 : f64
          %34 = tensor.insert %33 into %arg14[%arg7] : tensor<200xf64>
          %35 = arith.addi %arg15, %c1 : index
          scf.yield %34, %35 : tensor<200xf64>, index
        }
        scf.yield %29#0, %29#1 : tensor<200xf64>, index
      }
      scf.yield %28#0, %28#1 : tensor<200xf64>, index
    }
    %19 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel"]} ins(%15, %18#0 : tensor<200xf64>, tensor<200xf64>) outs(%15 : tensor<200xf64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %28 = arith.addf %arg7, %arg8 : f64
      linalg.yield %28 : f64
    } -> tensor<200xf64>
    %20 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["reduction"]} ins(%19 : tensor<200xf64>) outs(%cst_0 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %28 = arith.mulf %cst_5, %arg5 : f64
      %29 = arith.mulf %28, %arg5 : f64
      %30 = arith.mulf %29, %arg7 : f64
      %31 = arith.addf %30, %arg8 : f64
      linalg.yield %31 : f64
    } -> tensor<f64>
    %21 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["reduction"]} ins(%1 : tensor<200xf64>) outs(%cst_0 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %28 = arith.sitofp %arg6 : i64 to f64
      %29 = arith.mulf %28, %arg7 : f64
      %30 = arith.addf %29, %arg8 : f64
      linalg.yield %30 : f64
    } -> tensor<f64>
    %22 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = []} ins(%20, %21 : tensor<f64>, tensor<f64>) outs(%20 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %28 = arith.subf %arg7, %arg8 : f64
      linalg.yield %28 : f64
    } -> tensor<f64>
    %23 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["reduction"]} ins(%arg0 : tensor<200xf64>) outs(%cst_0 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %28 = math.exp %arg7 : f64
      %29 = arith.addf %28, %arg8 : f64
      linalg.yield %29 : f64
    } -> tensor<f64>
    %24 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = []} ins(%23 : tensor<f64>) outs(%23 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
      %28 = math.log %arg7 : f64
      linalg.yield %28 : f64
    } -> tensor<f64>
    %25 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = []} ins(%cst, %24 : tensor<f64>, tensor<f64>) outs(%cst : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %28 = arith.mulf %arg7, %arg8 : f64
      linalg.yield %28 : f64
    } -> tensor<f64>
    %26 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = []} ins(%14, %25 : tensor<f64>, tensor<f64>) outs(%14 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %28 = arith.subf %arg7, %arg8 : f64
      linalg.yield %28 : f64
    } -> tensor<f64>
    %27 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = []} ins(%26, %22 : tensor<f64>, tensor<f64>) outs(%26 : tensor<f64>) {
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
      %28 = arith.addf %arg7, %arg8 : f64
      linalg.yield %28 : f64
    } -> tensor<f64>
    return %27 : tensor<f64>
  }
}

