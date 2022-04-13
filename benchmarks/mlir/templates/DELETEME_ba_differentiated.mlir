// This is an experiment with the comprehensive module bufferization along with BA.
// The system for converting scf.if ops to use DPS is currently not developed,
// so that part is modified by hand.
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> ((d0 + 1) mod 3)>
#map3 = affine_map<(d0) -> ((d0 + 2) mod 3)>
module {
  func private @print_memref_f64(tensor<*xf64>) attributes {llvm.emit_c_interface}
  func @mlir_compute_reproj_error(%arg0: tensor<11xf64>, %arg1: tensor<3xf64>, %arg2: f64, %arg3: tensor<2xf64>, %arg4: tensor<2xf64>) -> tensor<2xf64> {
    %cst = arith.constant 1.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<2xf64>
    %c2 = arith.constant 2 : index
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<3xf64>
    %cst_3 = arith.constant 0.000000e+00 : f64
    %c6 = arith.constant 6 : index
    %0 = tensor.extract_slice %arg0[0] [3] [1] : tensor<11xf64> to tensor<3xf64>
    %1 = tensor.extract_slice %arg0[3] [3] [1] : tensor<11xf64> to tensor<3xf64>
    %2 = tensor.extract_slice %arg0[9] [2] [1] : tensor<11xf64> to tensor<2xf64>
    %3 = tensor.extract %arg0[%c6] : tensor<11xf64>
    %4 = tensor.extract_slice %arg0[7] [2] [1] : tensor<11xf64> to tensor<2xf64>
    %5 = arith.subf %arg1, %1 : tensor<3xf64>
    %6 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%0 : tensor<3xf64>) outs(%cst_0 : tensor<f64>) {
    ^bb0(%arg5: f64, %arg6: f64):
      %26 = arith.mulf %arg5, %arg5 : f64
      %27 = arith.addf %26, %arg6 : f64
      linalg.yield %27 : f64
    } -> tensor<f64>
    %7 = tensor.extract %6[] : tensor<f64>
    %8 = linalg.init_tensor [3] : tensor<3xf64>
    %9 = arith.cmpf one, %7, %cst_3 : f64
    %10 = scf.if %9 -> (tensor<3xf64>) {
      %26 = math.sqrt %7 : f64
      %27 = math.cos %26 : f64
      %28 = math.sin %26 : f64
      %29 = arith.divf %cst, %26 : f64
      %30 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%0 : tensor<3xf64>) outs(%cst_2 : tensor<3xf64>) {
      ^bb0(%arg5: f64, %arg6: f64):
        %41 = arith.mulf %arg5, %29 : f64
        linalg.yield %41 : f64
      } -> tensor<3xf64>
      %31 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0], iterator_types = ["parallel"]} ins(%30, %5, %30, %5 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst_2 : tensor<3xf64>) {
      ^bb0(%arg5: f64, %arg6: f64, %arg7: f64, %arg8: f64, %arg9: f64):
        %41 = arith.mulf %arg5, %arg6 : f64
        %42 = arith.mulf %arg7, %arg8 : f64
        %43 = arith.subf %41, %42 : f64
        linalg.yield %43 : f64
      } -> tensor<3xf64>
      %32 = linalg.dot ins(%30, %5 : tensor<3xf64>, tensor<3xf64>) outs(%cst_0 : tensor<f64>) -> tensor<f64>
      %33 = tensor.extract %32[] : tensor<f64>
      %34 = arith.subf %cst, %27 : f64
      %35 = arith.mulf %33, %34 : f64
      %36 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%5 : tensor<3xf64>) outs(%cst_2 : tensor<3xf64>) {
      ^bb0(%arg5: f64, %arg6: f64):
        %41 = arith.mulf %arg5, %27 : f64
        linalg.yield %41 : f64
      } -> tensor<3xf64>
      %37 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%31 : tensor<3xf64>) outs(%cst_2 : tensor<3xf64>) {
      ^bb0(%arg5: f64, %arg6: f64):
        %41 = arith.mulf %arg5, %28 : f64
        linalg.yield %41 : f64
      } -> tensor<3xf64>
      %38 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%30 : tensor<3xf64>) outs(%cst_2 : tensor<3xf64>) {
      ^bb0(%arg5: f64, %arg6: f64):
        %41 = arith.mulf %arg5, %35 : f64
        linalg.yield %41 : f64
      } -> tensor<3xf64>
      %39 = arith.addf %36, %37 : tensor<3xf64>
      %40 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%39, %38 : tensor<3xf64>, tensor<3xf64>) outs(%8 : tensor<3xf64>) {
      ^bb0(%arg5: f64, %arg6: f64, %arg7: f64):
        %41 = arith.addf %arg5, %arg6 : f64
        linalg.yield %41 : f64
      } -> tensor<3xf64>
      scf.yield %40 : tensor<3xf64>
    } else {
      %26 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0], iterator_types = ["parallel"]} ins(%0, %5, %0, %5 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst_2 : tensor<3xf64>) {
      ^bb0(%arg5: f64, %arg6: f64, %arg7: f64, %arg8: f64, %arg9: f64):
        %28 = arith.mulf %arg5, %arg6 : f64
        %29 = arith.mulf %arg7, %arg8 : f64
        %30 = arith.subf %28, %29 : f64
        linalg.yield %30 : f64
      } -> tensor<3xf64>
      %27 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%5, %26 : tensor<3xf64>, tensor<3xf64>) outs(%8 : tensor<3xf64>) {
      ^bb0(%arg5: f64, %arg6: f64, %arg7: f64):
        %28 = arith.addf %arg5, %arg6 : f64
        linalg.yield %28 : f64
      } -> tensor<3xf64>
      scf.yield %27 : tensor<3xf64>
    }
    %11 = tensor.extract_slice %10[0] [2] [1] : tensor<3xf64> to tensor<2xf64>
    %12 = tensor.extract %10[%c2] : tensor<3xf64>
    %13 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%11 : tensor<2xf64>) outs(%cst_1 : tensor<2xf64>) {
    ^bb0(%arg5: f64, %arg6: f64):
      %26 = arith.divf %arg5, %12 : f64
      linalg.yield %26 : f64
    } -> tensor<2xf64>
    %14 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%13 : tensor<2xf64>) outs(%cst_0 : tensor<f64>) {
    ^bb0(%arg5: f64, %arg6: f64):
      %26 = arith.mulf %arg5, %arg5 : f64
      %27 = arith.addf %26, %arg6 : f64
      linalg.yield %27 : f64
    } -> tensor<f64>
    %15 = tensor.extract %14[] : tensor<f64>
    %16 = tensor.extract %2[%c0] : tensor<2xf64>
    %17 = tensor.extract %2[%c1] : tensor<2xf64>
    %18 = arith.mulf %16, %15 : f64
    %19 = arith.mulf %17, %15 : f64
    %20 = arith.mulf %19, %15 : f64
    %21 = arith.addf %18, %20 : f64
    %22 = arith.addf %21, %cst : f64
    %23 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%13 : tensor<2xf64>) outs(%arg4 : tensor<2xf64>) {
    ^bb0(%arg5: f64, %arg6: f64):
      %26 = arith.mulf %arg5, %22 : f64
      linalg.yield %26 : f64
    } -> tensor<2xf64>
    %24 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%4 : tensor<2xf64>) outs(%23 : tensor<2xf64>) {
    ^bb0(%arg5: f64, %arg6: f64):
      %26 = arith.mulf %arg6, %3 : f64
      %27 = arith.addf %26, %arg5 : f64
      linalg.yield %27 : f64
    } -> tensor<2xf64>
    %25 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%arg3 : tensor<2xf64>) outs(%24 : tensor<2xf64>) {
    ^bb0(%arg5: f64, %arg6: f64):
      %26 = arith.subf %arg6, %arg5 : f64
      %27 = arith.mulf %arg2, %26 : f64
      linalg.yield %27 : f64
    } -> tensor<2xf64>
    return %25 : tensor<2xf64>
  }
  func @__grad_mlir_compute_reproj_error(%arg0: tensor<11xf64>, %arg1: tensor<3xf64>, %arg2: f64, %arg3: tensor<2xf64>, %arg4: tensor<2xf64>, %arg5: tensor<11xf64>, %arg6: tensor<3xf64>, %arg7: f64, %arg8: tensor<2xf64>) -> (tensor<11xf64>, tensor<3xf64>, f64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<3xf64>
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 5.000000e-01 : f64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<2xf64>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_4 = arith.constant 1.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %0 = tensor.extract_slice %arg0[0] [3] [1] : tensor<11xf64> to tensor<3xf64>
    %1 = tensor.extract_slice %arg0[3] [3] [1] : tensor<11xf64> to tensor<3xf64>
    %2 = tensor.extract_slice %arg0[9] [2] [1] : tensor<11xf64> to tensor<2xf64>
    %3 = tensor.extract_slice %arg0[7] [2] [1] : tensor<11xf64> to tensor<2xf64>
    %4 = arith.subf %arg1, %1 : tensor<3xf64>
    %5 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%0 : tensor<3xf64>) outs(%cst_3 : tensor<f64>) {
    ^bb0(%arg9: f64, %arg10: f64):
      %81 = arith.mulf %arg9, %arg9 : f64
      %82 = arith.addf %81, %arg10 : f64
      linalg.yield %82 : f64
    } -> tensor<f64>
    %6 = tensor.extract %5[] : tensor<f64>
    %7 = linalg.init_tensor [3] : tensor<3xf64>
    %8 = arith.cmpf one, %6, %cst_0 : f64
    %9 = scf.if %8 -> (tensor<3xf64>) {
      %81 = math.sqrt %6 : f64
      %82 = math.cos %81 : f64
      %83 = math.sin %81 : f64
      %84 = arith.divf %cst_4, %81 : f64
      %85 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%0 : tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %96 = arith.mulf %arg9, %84 : f64
        linalg.yield %96 : f64
      } -> tensor<3xf64>
      %86 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0], iterator_types = ["parallel"]} ins(%85, %4, %85, %4 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64):
        %96 = arith.mulf %arg9, %arg10 : f64
        %97 = arith.mulf %arg11, %arg12 : f64
        %98 = arith.subf %96, %97 : f64
        linalg.yield %98 : f64
      } -> tensor<3xf64>
      %87 = linalg.dot ins(%85, %4 : tensor<3xf64>, tensor<3xf64>) outs(%cst_3 : tensor<f64>) -> tensor<f64>
      %88 = tensor.extract %87[] : tensor<f64>
      %89 = arith.subf %cst_4, %82 : f64
      %90 = arith.mulf %88, %89 : f64
      %91 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%4 : tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %96 = arith.mulf %arg9, %82 : f64
        linalg.yield %96 : f64
      } -> tensor<3xf64>
      %92 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%86 : tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %96 = arith.mulf %arg9, %83 : f64
        linalg.yield %96 : f64
      } -> tensor<3xf64>
      %93 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%85 : tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %96 = arith.mulf %arg9, %90 : f64
        linalg.yield %96 : f64
      } -> tensor<3xf64>
      %94 = arith.addf %91, %92 : tensor<3xf64>
      %95 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%94, %93 : tensor<3xf64>, tensor<3xf64>) outs(%7 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %96 = arith.addf %arg9, %arg10 : f64
        linalg.yield %96 : f64
      } -> tensor<3xf64>
      scf.yield %95 : tensor<3xf64>
    } else {
      %81 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0], iterator_types = ["parallel"]} ins(%0, %4, %0, %4 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64):
        %83 = arith.mulf %arg9, %arg10 : f64
        %84 = arith.mulf %arg11, %arg12 : f64
        %85 = arith.subf %83, %84 : f64
        linalg.yield %85 : f64
      } -> tensor<3xf64>
      %82 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%4, %81 : tensor<3xf64>, tensor<3xf64>) outs(%7 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %83 = arith.addf %arg9, %arg10 : f64
        linalg.yield %83 : f64
      } -> tensor<3xf64>
      scf.yield %82 : tensor<3xf64>
    }
    %10 = tensor.extract_slice %9[0] [2] [1] : tensor<3xf64> to tensor<2xf64>
    %11 = tensor.extract %9[%c2] : tensor<3xf64>
    %12 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%10 : tensor<2xf64>) outs(%cst_2 : tensor<2xf64>) {
    ^bb0(%arg9: f64, %arg10: f64):
      %81 = arith.divf %arg9, %11 : f64
      linalg.yield %81 : f64
    } -> tensor<2xf64>
    %13 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%12 : tensor<2xf64>) outs(%cst_3 : tensor<f64>) {
    ^bb0(%arg9: f64, %arg10: f64):
      %81 = arith.mulf %arg9, %arg9 : f64
      %82 = arith.addf %81, %arg10 : f64
      linalg.yield %82 : f64
    } -> tensor<f64>
    %14 = tensor.extract %13[] : tensor<f64>
    %15 = tensor.extract %2[%c0] : tensor<2xf64>
    %16 = tensor.extract %2[%c1] : tensor<2xf64>
    %17 = arith.mulf %15, %14 : f64
    %18 = arith.mulf %16, %14 : f64
    %19 = arith.mulf %18, %14 : f64
    %20 = arith.addf %17, %19 : f64
    %21 = arith.addf %20, %cst_4 : f64
    %22 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel"]} ins(%arg3, %arg8 : tensor<2xf64>, tensor<2xf64>) outs(%cst_3 : tensor<f64>) {
    ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
      %81 = arith.subf %arg11, %arg9 : f64
      %82 = arith.mulf %arg10, %81 : f64
      %83 = arith.addf %82, %arg11 : f64
      linalg.yield %83 : f64
    } -> tensor<f64>
    %23 = tensor.extract %22[] : tensor<f64>
    %24 = arith.addf %23, %arg7 : f64
    %25 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel"]} ins(%3, %arg8 : tensor<2xf64>, tensor<2xf64>) outs(%cst_3 : tensor<f64>) {
    ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
      %81 = arith.mulf %arg10, %arg11 : f64
      %82 = arith.addf %81, %arg11 : f64
      linalg.yield %82 : f64
    } -> tensor<f64>
    %26 = tensor.extract %25[] : tensor<f64>
    %27 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel"]} ins(%12, %arg8 : tensor<2xf64>, tensor<2xf64>) outs(%cst_3 : tensor<f64>) {
    ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
      %81 = arith.mulf %arg10, %arg9 : f64
      %82 = arith.addf %81, %arg11 : f64
      linalg.yield %82 : f64
    } -> tensor<f64>
    %28 = tensor.extract %27[] : tensor<f64>
    %29 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%12, %arg8 : tensor<2xf64>, tensor<2xf64>) outs(%cst_2 : tensor<2xf64>) {
    ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
      %81 = arith.mulf %arg10, %21 : f64
      linalg.yield %81 : f64
    } -> tensor<2xf64>
    %30 = arith.mulf %28, %14 : f64
    %31 = arith.mulf %28, %18 : f64
    %32 = arith.mulf %30, %14 : f64
    %33 = arith.mulf %30, %16 : f64
    %34 = arith.addf %33, %31 : f64
    %35 = arith.mulf %28, %14 : f64
    %36 = arith.mulf %28, %15 : f64
    %37 = arith.addf %36, %34 : f64
    %38 = linalg.init_tensor [2] : tensor<2xf64>
    %39 = linalg.fill(%cst_0, %38) : f64, tensor<2xf64> -> tensor<2xf64> 
    %40 = tensor.insert %32 into %39[%c1] : tensor<2xf64>
    %41 = tensor.extract %40[%c0] : tensor<2xf64>
    %42 = arith.addf %41, %35 : f64
    %43 = tensor.insert %42 into %40[%c0] : tensor<2xf64>
    %44 = linalg.init_tensor [] : tensor<f64>
    %45 = linalg.fill(%cst_0, %44) : f64, tensor<f64> -> tensor<f64> 
    %46 = tensor.insert %37 into %45[] : tensor<f64>
    %47 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel"]} ins(%12, %46 : tensor<2xf64>, tensor<f64>) outs(%cst_2 : tensor<2xf64>) {
    ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
      %81 = arith.mulf %arg10, %arg9 : f64
      %82 = arith.mulf %arg10, %arg9 : f64
      %83 = arith.addf %82, %81 : f64
      linalg.yield %83 : f64
    } -> tensor<2xf64>
    %48 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%47 : tensor<2xf64>) outs(%29 : tensor<2xf64>) {
    ^bb0(%arg9: f64, %arg10: f64):
      %81 = arith.addf %arg9, %arg10 : f64
      linalg.yield %81 : f64
    } -> tensor<2xf64>
    %49 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel"]} ins(%10, %48 : tensor<2xf64>, tensor<2xf64>) outs(%cst_3 : tensor<f64>) {
    ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
      %81 = arith.mulf %arg10, %arg9 : f64
      %82 = arith.negf %81 : f64
      %83 = arith.mulf %11, %11 : f64
      %84 = arith.divf %82, %83 : f64
      %85 = arith.addf %84, %arg11 : f64
      linalg.yield %85 : f64
    } -> tensor<f64>
    %50 = tensor.extract %49[] : tensor<f64>
    %51 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%10, %48 : tensor<2xf64>, tensor<2xf64>) outs(%cst_2 : tensor<2xf64>) {
    ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
      %81 = arith.divf %arg10, %11 : f64
      linalg.yield %81 : f64
    } -> tensor<2xf64>
    %52 = linalg.init_tensor [3] : tensor<3xf64>
    %53 = linalg.fill(%cst_0, %52) : f64, tensor<3xf64> -> tensor<3xf64> 
    %54 = tensor.insert %50 into %53[%c2] : tensor<3xf64>
    %55 = tensor.extract_slice %54[0] [2] [1] : tensor<3xf64> to tensor<2xf64>
    %56 = arith.addf %55, %51 : tensor<2xf64>
    %57 = tensor.insert_slice %56 into %54[0] [2] [1] : tensor<2xf64> into tensor<3xf64>
    %res0_space = linalg.init_tensor [3] : tensor<3xf64>
    %res1_space = linalg.init_tensor [3] : tensor<3xf64>
    %58:3 = scf.if %8 -> (tensor<3xf64>, f64, tensor<3xf64>) {
      %81 = math.sqrt %6 : f64
      %82 = math.cos %81 : f64
      %83 = math.sin %81 : f64
      %84 = arith.divf %cst_4, %81 : f64
      %85 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%0 : tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %172 = arith.mulf %arg9, %84 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %86 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0], iterator_types = ["parallel"]} ins(%85, %4, %85, %4 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64):
        %172 = arith.mulf %arg9, %arg10 : f64
        %173 = arith.mulf %arg11, %arg12 : f64
        %174 = arith.subf %172, %173 : f64
        linalg.yield %174 : f64
      } -> tensor<3xf64>
      %87 = linalg.dot ins(%85, %4 : tensor<3xf64>, tensor<3xf64>) outs(%cst_3 : tensor<f64>) -> tensor<f64>
      %88 = tensor.extract %87[] : tensor<f64>
      %89 = arith.subf %cst_4, %82 : f64
      %90 = arith.mulf %88, %89 : f64
      %91 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel"]} ins(%85, %57 : tensor<3xf64>, tensor<3xf64>) outs(%cst_3 : tensor<f64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %arg9 : f64
        %173 = arith.addf %172, %arg11 : f64
        linalg.yield %173 : f64
      } -> tensor<f64>
      %92 = tensor.extract %91[] : tensor<f64>
      %93 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%85, %57 : tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %90 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %94 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%86, %57 : tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %83 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %95 = arith.mulf %92, %89 : f64
      %96 = linalg.init_tensor [] : tensor<f64>
      %97 = linalg.fill(%cst_0, %96) : f64, tensor<f64> -> tensor<f64> 
      %98 = tensor.insert %95 into %97[] : tensor<f64>
      %99 = linalg.generic {doc = "Copy and scalar multiplication", indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel"], library_call = "sdot_grad_first"} ins(%98, %4 : tensor<f64>, tensor<3xf64>) outs(%85 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg9, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %100 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%99 : tensor<3xf64>) outs(%93 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %172 = arith.addf %arg9, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %101 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0, #map2], iterator_types = ["parallel"]} ins(%85, %4, %85, %4, %94 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):
        %172 = arith.mulf %arg13, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %102 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%101 : tensor<3xf64>) outs(%100 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %172 = arith.addf %arg9, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %103 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0, #map3], iterator_types = ["parallel"]} ins(%85, %4, %85, %4, %94 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):
        %172 = arith.negf %arg13 : f64
        %173 = arith.mulf %172, %arg12 : f64
        linalg.yield %173 : f64
      } -> tensor<3xf64>
      %104 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%103 : tensor<3xf64>) outs(%102 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %172 = arith.addf %arg9, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %105 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%0, %104 : tensor<3xf64>, tensor<3xf64>) outs(%res0_space : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %84 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %106 = math.sqrt %6 : f64
      %107 = math.cos %106 : f64
      %108 = math.sin %106 : f64
      %109 = arith.divf %cst_4, %106 : f64
      %110 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%0 : tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %172 = arith.mulf %arg9, %109 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %111 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0], iterator_types = ["parallel"]} ins(%110, %4, %110, %4 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64):
        %172 = arith.mulf %arg9, %arg10 : f64
        %173 = arith.mulf %arg11, %arg12 : f64
        %174 = arith.subf %172, %173 : f64
        linalg.yield %174 : f64
      } -> tensor<3xf64>
      %112 = linalg.dot ins(%110, %4 : tensor<3xf64>, tensor<3xf64>) outs(%cst_3 : tensor<f64>) -> tensor<f64>
      %113 = tensor.extract %112[] : tensor<f64>
      %114 = arith.subf %cst_4, %107 : f64
      %115 = arith.mulf %113, %114 : f64
      %116 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel"]} ins(%110, %57 : tensor<3xf64>, tensor<3xf64>) outs(%cst_3 : tensor<f64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %arg9 : f64
        %173 = arith.addf %172, %arg11 : f64
        linalg.yield %173 : f64
      } -> tensor<f64>
      %117 = tensor.extract %116[] : tensor<f64>
      %118 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%110, %57 : tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %115 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %119 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel"]} ins(%111, %57 : tensor<3xf64>, tensor<3xf64>) outs(%cst_3 : tensor<f64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %arg9 : f64
        %173 = arith.addf %172, %arg11 : f64
        linalg.yield %173 : f64
      } -> tensor<f64>
      %120 = tensor.extract %119[] : tensor<f64>
      %121 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%111, %57 : tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %108 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %122 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel"]} ins(%4, %57 : tensor<3xf64>, tensor<3xf64>) outs(%cst_3 : tensor<f64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %arg9 : f64
        %173 = arith.addf %172, %arg11 : f64
        linalg.yield %173 : f64
      } -> tensor<f64>
      %123 = tensor.extract %122[] : tensor<f64>
      %124 = arith.mulf %117, %114 : f64
      %125 = arith.mulf %117, %113 : f64
      %126 = arith.negf %125 : f64
      %127 = arith.addf %126, %123 : f64
      %128 = linalg.init_tensor [] : tensor<f64>
      %129 = linalg.fill(%cst_0, %128) : f64, tensor<f64> -> tensor<f64> 
      %130 = tensor.insert %124 into %129[] : tensor<f64>
      %131 = linalg.generic {doc = "Copy and scalar multiplication", indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel"], library_call = "sdot_grad_first"} ins(%130, %4 : tensor<f64>, tensor<3xf64>) outs(%110 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg9, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %132 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%131 : tensor<3xf64>) outs(%118 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %172 = arith.addf %arg9, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %133 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0, #map2], iterator_types = ["parallel"]} ins(%110, %4, %110, %4, %121 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):
        %172 = arith.mulf %arg13, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %134 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%133 : tensor<3xf64>) outs(%132 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %172 = arith.addf %arg9, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %135 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0, #map3], iterator_types = ["parallel"]} ins(%110, %4, %110, %4, %121 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):
        %172 = arith.negf %arg13 : f64
        %173 = arith.mulf %172, %arg12 : f64
        linalg.yield %173 : f64
      } -> tensor<3xf64>
      %136 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%135 : tensor<3xf64>) outs(%134 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %172 = arith.addf %arg9, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %137 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel"]} ins(%0, %136 : tensor<3xf64>, tensor<3xf64>) outs(%cst_3 : tensor<f64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %arg9 : f64
        %173 = arith.addf %172, %arg11 : f64
        linalg.yield %173 : f64
      } -> tensor<f64>
      %138 = tensor.extract %137[] : tensor<f64>
      %139 = arith.negf %138 : f64
      %140 = arith.mulf %106, %106 : f64
      %141 = arith.divf %139, %140 : f64
      %142 = math.cos %106 : f64
      %143 = arith.mulf %142, %120 : f64
      %144 = arith.addf %143, %141 : f64
      %145 = math.sin %106 : f64
      %146 = arith.mulf %145, %127 : f64
      %147 = arith.negf %146 : f64
      %148 = arith.addf %147, %144 : f64
      %149 = arith.mulf %148, %cst_1 : f64
      %150 = arith.divf %149, %106 : f64
      %151 = math.sqrt %6 : f64
      %152 = math.cos %151 : f64
      %153 = math.sin %151 : f64
      %154 = arith.divf %cst_4, %151 : f64
      %155 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%0 : tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %172 = arith.mulf %arg9, %154 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %156 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0], iterator_types = ["parallel"]} ins(%155, %4, %155, %4 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64):
        %172 = arith.mulf %arg9, %arg10 : f64
        %173 = arith.mulf %arg11, %arg12 : f64
        %174 = arith.subf %172, %173 : f64
        linalg.yield %174 : f64
      } -> tensor<3xf64>
      %157 = arith.subf %cst_4, %152 : f64
      %158 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel"]} ins(%155, %57 : tensor<3xf64>, tensor<3xf64>) outs(%cst_3 : tensor<f64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %arg9 : f64
        %173 = arith.addf %172, %arg11 : f64
        linalg.yield %173 : f64
      } -> tensor<f64>
      %159 = tensor.extract %158[] : tensor<f64>
      %160 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%156, %57 : tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %153 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %161 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%4, %57 : tensor<3xf64>, tensor<3xf64>) outs(%res1_space : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg10, %152 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %162 = arith.mulf %159, %157 : f64
      %163 = linalg.init_tensor [] : tensor<f64>
      %164 = linalg.fill(%cst_0, %163) : f64, tensor<f64> -> tensor<f64> 
      %165 = tensor.insert %162 into %164[] : tensor<f64>
      %166 = linalg.generic {doc = "Copy and scalar multiplication", indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel"], library_call = "sdot_grad_second"} ins(%165, %155 : tensor<f64>, tensor<3xf64>) outs(%4 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %172 = arith.mulf %arg9, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %167 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%166 : tensor<3xf64>) outs(%161 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %172 = arith.addf %arg9, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %168 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0, #map3], iterator_types = ["parallel"]} ins(%155, %4, %155, %4, %160 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):
        %172 = arith.mulf %arg13, %arg9 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %169 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%168 : tensor<3xf64>) outs(%167 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %172 = arith.addf %arg9, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      %170 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0, #map2], iterator_types = ["parallel"]} ins(%155, %4, %155, %4, %160 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):
        %172 = arith.negf %arg13 : f64
        %173 = arith.mulf %172, %arg11 : f64
        linalg.yield %173 : f64
      } -> tensor<3xf64>
      %171 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%170 : tensor<3xf64>) outs(%169 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %172 = arith.addf %arg9, %arg10 : f64
        linalg.yield %172 : f64
      } -> tensor<3xf64>
      scf.yield %105, %150, %171 : tensor<3xf64>, f64, tensor<3xf64>
    } else {
      %81 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0, #map2], iterator_types = ["parallel"]} ins(%0, %4, %0, %4, %57 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%res0_space : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):
        %88 = arith.mulf %arg13, %arg10 : f64
        linalg.yield %88 : f64
      } -> tensor<3xf64>
      %82 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0, #map3], iterator_types = ["parallel"]} ins(%0, %4, %0, %4, %57 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):
        %88 = arith.negf %arg13 : f64
        %89 = arith.mulf %88, %arg12 : f64
        linalg.yield %89 : f64
      } -> tensor<3xf64>
      %83 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%82 : tensor<3xf64>) outs(%81 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %88 = arith.addf %arg9, %arg10 : f64
        linalg.yield %88 : f64
      } -> tensor<3xf64>
      %84 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0, #map3], iterator_types = ["parallel"]} ins(%0, %4, %0, %4, %57 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):
        %88 = arith.mulf %arg13, %arg9 : f64
        linalg.yield %88 : f64
      } -> tensor<3xf64>
      %85 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%84, %57 : tensor<3xf64>, tensor<3xf64>) outs(%res1_space : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
        %88 = arith.addf %arg9, %arg10 : f64
        linalg.yield %88 : f64
      } -> tensor<3xf64>
      %86 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map0, #map2], iterator_types = ["parallel"]} ins(%0, %4, %0, %4, %57 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) outs(%cst : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):
        %88 = arith.negf %arg13 : f64
        %89 = arith.mulf %88, %arg11 : f64
        linalg.yield %89 : f64
      } -> tensor<3xf64>
      %87 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%86 : tensor<3xf64>) outs(%85 : tensor<3xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):
        %88 = arith.addf %arg9, %arg10 : f64
        linalg.yield %88 : f64
      } -> tensor<3xf64>
      scf.yield %83, %cst_0, %87 : tensor<3xf64>, f64, tensor<3xf64>
    }
    %59 = linalg.init_tensor [] : tensor<f64>
    %60 = linalg.fill(%cst_0, %59) : f64, tensor<f64> -> tensor<f64> 
    %61 = tensor.insert %58#1 into %60[] : tensor<f64>
    %62 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel"]} ins(%0, %61 : tensor<3xf64>, tensor<f64>) outs(%cst : tensor<3xf64>) {
    ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):
      %81 = arith.mulf %arg10, %arg9 : f64
      %82 = arith.mulf %arg10, %arg9 : f64
      %83 = arith.addf %82, %81 : f64
      linalg.yield %83 : f64
    } -> tensor<3xf64>
    %63 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%62 : tensor<3xf64>) outs(%58#0 : tensor<3xf64>) {
    ^bb0(%arg9: f64, %arg10: f64):
      %81 = arith.addf %arg9, %arg10 : f64
      linalg.yield %81 : f64
    } -> tensor<3xf64>
    %64 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%58#2 : tensor<3xf64>) outs(%arg6 : tensor<3xf64>) {
    ^bb0(%arg9: f64, %arg10: f64):
      %81 = arith.addf %arg9, %arg10 : f64
      linalg.yield %81 : f64
    } -> tensor<3xf64>
    %65 = arith.negf %58#2 : tensor<3xf64>
    %66 = tensor.extract_slice %arg5[7] [2] [1] : tensor<11xf64> to tensor<2xf64>
    %67 = arith.addf %66, %arg8 : tensor<2xf64>
    %68 = tensor.insert_slice %67 into %arg5[7] [2] [1] : tensor<2xf64> into tensor<11xf64>
    %69 = tensor.extract %68[%c6] : tensor<11xf64>
    %70 = arith.addf %69, %26 : f64
    %71 = tensor.insert %70 into %68[%c6] : tensor<11xf64>
    %72 = tensor.extract_slice %71[9] [2] [1] : tensor<11xf64> to tensor<2xf64>
    %73 = arith.addf %72, %43 : tensor<2xf64>
    %74 = tensor.insert_slice %73 into %71[9] [2] [1] : tensor<2xf64> into tensor<11xf64>
    %75 = tensor.extract_slice %74[3] [3] [1] : tensor<11xf64> to tensor<3xf64>
    %76 = arith.addf %75, %65 : tensor<3xf64>
    %77 = tensor.insert_slice %76 into %74[3] [3] [1] : tensor<3xf64> into tensor<11xf64>
    %78 = tensor.extract_slice %77[0] [3] [1] : tensor<11xf64> to tensor<3xf64>
    %79 = arith.addf %78, %63 : tensor<3xf64>
    %80 = tensor.insert_slice %79 into %77[0] [3] [1] : tensor<3xf64> into tensor<11xf64>
    return %80, %64, %24 : tensor<11xf64>, tensor<3xf64>, f64
  }
  func @mlir_compute_zach_weight_error(%arg0: f64) -> f64 {
    %cst = arith.constant 1.000000e+00 : f64
    %0 = arith.mulf %arg0, %arg0 : f64
    %1 = arith.subf %cst, %0 : f64
    return %1 : f64
  }
  func @__grad_mlir_compute_zach_weight_error(%arg0: f64) -> f64 {
    %cst = arith.constant 1.000000e+00 : f64
    %0 = arith.negf %cst : f64
    %1 = arith.mulf %0, %arg0 : f64
    %2 = arith.mulf %0, %arg0 : f64
    %3 = arith.addf %2, %1 : f64
    return %3 : f64
  }
  func @lagrad_compute_reproj_error(%arg0: tensor<11xf64>, %arg1: tensor<11xf64>, %arg2: tensor<3xf64>, %arg3: tensor<3xf64>, %arg4: f64, %arg5: f64, %arg6: tensor<2xf64>, %arg7: tensor<2xf64>) -> (tensor<11xf64>, tensor<3xf64>, f64) {
    %0 = linalg.init_tensor [2] : tensor<2xf64>
    %1:3 = call @__grad_mlir_compute_reproj_error(%arg0, %arg2, %arg4, %arg6, %0, %arg1, %arg3, %arg5, %arg7) : (tensor<11xf64>, tensor<3xf64>, f64, tensor<2xf64>, tensor<2xf64>, tensor<11xf64>, tensor<3xf64>, f64, tensor<2xf64>) -> (tensor<11xf64>, tensor<3xf64>, f64)
    return %1#0, %1#1, %1#2 : tensor<11xf64>, tensor<3xf64>, f64
  }
  func @lagrad_compute_w_error(%arg0: f64) -> f64 {
    %0 = call @__grad_mlir_compute_zach_weight_error(%arg0) : (f64) -> f64
    return %0 : f64
  }
}

