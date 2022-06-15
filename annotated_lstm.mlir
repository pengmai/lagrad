#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1, d2)[s0] -> (d0 * 28 + s0 + d1 * 14 + d2)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0)>
module  {
  func private @msigmoid(%arg0: f64) -> f64 {
    %cst = arith.constant 1.000000e+00 : f64
    %0 = arith.negf %arg0 : f64
    %1 = math.exp %0 : f64
    %2 = arith.addf %cst, %1 : f64
    %3 = arith.divf %cst, %2 : f64
    return %3 : f64
  }
  func private @__grad_msigmoid_arg0(%arg0: f64, %arg1: f64) -> f64 {
    %cst = arith.constant 1.000000e+00 : f64
    %0 = arith.negf %arg0 : f64
    %1 = math.exp %0 : f64
    %2 = arith.addf %cst, %1 : f64
    %3 = arith.mulf %arg1, %cst : f64
    %4 = arith.negf %3 : f64
    %5 = arith.mulf %2, %2 : f64
    %6 = arith.divf %4, %5 : f64
    %7 = arith.mulf %6, %1 : f64
    %8 = arith.negf %7 : f64
    return %8 : f64
  }
  func @mlogsumexp(%arg0: tensor<14xf64>) -> f64 {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f64>
    %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<14xf64>) outs(%cst_0 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):  // no predecessors
      %4 = math.exp %arg1 : f64
      %5 = arith.addf %4, %arg2 : f64
      linalg.yield %5 : f64
    } -> tensor<f64>
    %1 = tensor.extract %0[] : tensor<f64>
    %2 = arith.addf %1, %cst : f64
    %3 = math.log %2 : f64
    return %3 : f64
  }
  func @__grad_mlogsumexp_arg0(%arg0: tensor<14xf64>, %arg1: f64) -> tensor<14xf64> {
    %cst = arith.constant dense<0.000000e+00> : tensor<14xf64>
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<f64>
    %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<14xf64>) outs(%cst_2 : tensor<f64>) {
    ^bb0(%arg2: f64, %arg3: f64):  // no predecessors
      %8 = math.exp %arg2 : f64
      %9 = arith.addf %8, %arg3 : f64
      linalg.yield %9 : f64
    } -> tensor<f64>
    %1 = tensor.extract %0[] : tensor<f64>
    %2 = arith.addf %1, %cst_1 : f64
    %3 = arith.divf %arg1, %2 : f64
    %4 = linalg.init_tensor [] : tensor<f64>
    %5 = linalg.fill(%cst_0, %4) : f64, tensor<f64> -> tensor<f64> 
    %6 = tensor.insert %3 into %5[] : tensor<f64>
    %7 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel"]} ins(%arg0, %6 : tensor<14xf64>, tensor<f64>) outs(%cst : tensor<14xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
      %8 = math.exp %arg2 : f64
      %9 = arith.mulf %arg3, %8 : f64
      linalg.yield %9 : f64
    } -> tensor<14xf64>
    return %7 : tensor<14xf64>
  }

  func @__grad_mlstm_objective(%arg0: tensor<2x2x4x14xf64>, %arg1: tensor<3x14xf64>, %arg2: tensor<2x2x14xf64>, %arg3: tensor<4x14xf64>) -> (tensor<2x2x4x14xf64>, tensor<3x14xf64>) {
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<14xf64>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<f64>
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c14 = arith.constant 14 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    // This is the state cache, looks to be storing the correct values.
    %0 = memref.alloc() : memref<3x2x2x14xf64>
    %1:3 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %cst_0, %arg6 = %c0, %arg7 = %arg2) -> (f64, index, tensor<2x2x14xf64>) {
      %15 = memref.subview %0[%arg4, 0, 0, 0] [1, 2, 2, 14] [1, 1, 1, 1] : memref<3x2x2x14xf64> to memref<2x2x14xf64, #map2>
      %16 = memref.buffer_cast %arg7 : memref<2x2x14xf64>
      linalg.copy(%16, %15) : memref<2x2x14xf64>, memref<2x2x14xf64, #map2>
      %17 = tensor.extract_slice %arg3[%arg4, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
      %18 = tensor.extract_slice %arg1[0, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
      %19 = arith.mulf %17, %18 : tensor<14xf64>
      %20:2 = scf.for %arg8 = %c0 to %c2 step %c1 iter_args(%arg9 = %19, %arg10 = %arg7) -> (tensor<14xf64>, tensor<2x2x14xf64>) {
        %32 = tensor.extract_slice %arg0[%arg8, 0, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %33 = tensor.extract_slice %arg0[%arg8, 1, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %34 = tensor.extract_slice %arg0[%arg8, 0, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %35 = tensor.extract_slice %arg0[%arg8, 1, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %36 = tensor.extract_slice %arg0[%arg8, 0, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %37 = tensor.extract_slice %arg0[%arg8, 1, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %38 = tensor.extract_slice %arg0[%arg8, 0, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %39 = tensor.extract_slice %arg0[%arg8, 1, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %40 = tensor.extract_slice %arg10[%arg8, 0, 0] [1, 1, 14] [1, 1, 1] : tensor<2x2x14xf64> to tensor<14xf64>
        %41 = tensor.extract_slice %arg10[%arg8, 1, 0] [1, 1, 14] [1, 1, 1] : tensor<2x2x14xf64> to tensor<14xf64>
        %42 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg9, %32, %33 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):  // no predecessors
          %50 = arith.mulf %arg11, %arg12 : f64
          %51 = arith.addf %50, %arg13 : f64
          %52 = call @msigmoid(%51) : (f64) -> f64
          linalg.yield %52 : f64
        } -> tensor<14xf64>
        %43 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%40, %34, %35 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):  // no predecessors
          %50 = arith.mulf %arg11, %arg12 : f64
          %51 = arith.addf %50, %arg13 : f64
          %52 = call @msigmoid(%51) : (f64) -> f64
          linalg.yield %52 : f64
        } -> tensor<14xf64>
        %44 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg9, %36, %37 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):  // no predecessors
          %50 = arith.mulf %arg11, %arg12 : f64
          %51 = arith.addf %50, %arg13 : f64
          %52 = call @msigmoid(%51) : (f64) -> f64
          linalg.yield %52 : f64
        } -> tensor<14xf64>
        %45 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%40, %38, %39 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64):  // no predecessors
          %50 = arith.mulf %arg11, %arg12 : f64
          %51 = arith.addf %50, %arg13 : f64
          %52 = math.tanh %51 : f64
          linalg.yield %52 : f64
        } -> tensor<14xf64>
        %46 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%41, %42, %43, %45 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64, %arg15: f64):  // no predecessors
          %50 = arith.mulf %arg11, %arg12 : f64
          %51 = arith.mulf %arg13, %arg14 : f64
          %52 = arith.addf %50, %51 : f64
          linalg.yield %52 : f64
        } -> tensor<14xf64>
        %47 = tensor.insert_slice %46 into %arg10[%arg8, 1, 0] [1, 1, 14] [1, 1, 1] : tensor<14xf64> into tensor<2x2x14xf64>
        %48 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%44, %46 : tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg11: f64, %arg12: f64, %arg13: f64):  // no predecessors
          %50 = math.tanh %arg12 : f64
          %51 = arith.mulf %arg11, %50 : f64
          linalg.yield %51 : f64
        } -> tensor<14xf64>
        %49 = tensor.insert_slice %48 into %47[%arg8, 0, 0] [1, 1, 14] [1, 1, 1] : tensor<14xf64> into tensor<2x2x14xf64>
        scf.yield %48, %49 : tensor<14xf64>, tensor<2x2x14xf64>
      }
      %21 = tensor.extract_slice %arg1[1, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
      %22 = tensor.extract_slice %arg1[2, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
      %23 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%20#0, %21, %22 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
      ^bb0(%arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
        %32 = arith.mulf %arg8, %arg9 : f64
        %33 = arith.addf %32, %arg10 : f64
        linalg.yield %33 : f64
      } -> tensor<14xf64>
      %24 = call @mlogsumexp(%23) : (tensor<14xf64>) -> f64
      %25 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%23 : tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
      ^bb0(%arg8: f64, %arg9: f64):  // no predecessors
        %32 = arith.subf %arg8, %24 : f64
        linalg.yield %32 : f64
      } -> tensor<14xf64>
      %26 = arith.addi %arg4, %c1 : index
      %27 = tensor.extract_slice %arg3[%26, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
      %28 = linalg.dot ins(%27, %25 : tensor<14xf64>, tensor<14xf64>) outs(%cst_3 : tensor<f64>) -> tensor<f64>
      %29 = tensor.extract %28[] : tensor<f64>
      %30 = arith.addf %29, %arg5 : f64
      %31 = arith.addi %arg6, %c14 : index
      scf.yield %30, %31, %20#1 : f64, index, tensor<2x2x14xf64>
    }
    %2 = arith.index_cast %1#1 : index to i64
    %3 = arith.sitofp %2 : i64 to f64
    %4 = arith.divf %cst_1, %3 : f64
    %5 = arith.negf %4 : f64

    //
    // ADJOINT STARTS HERE
    //
    // dmain_params
    %6 = linalg.init_tensor [2, 2, 4, 14] : tensor<2x2x4x14xf64>
    %7 = linalg.fill(%cst_0, %6) : f64, tensor<2x2x4x14xf64> -> tensor<2x2x4x14xf64>
    // I think this is dx
    %8 = linalg.init_tensor [4, 14] : tensor<4x14xf64>
    %9 = linalg.fill(%cst_0, %8) : f64, tensor<4x14xf64> -> tensor<4x14xf64>
    // Gradient space for extra params is initialized here
    %10 = linalg.init_tensor [3, 14] : tensor<3x14xf64>
    %11 = linalg.fill(%cst_0, %10) : f64, tensor<3x14xf64> -> tensor<3x14xf64>
    // This should be dstate
    %12 = linalg.init_tensor [2, 2, 14] : tensor<2x2x14xf64>
    %13 = linalg.fill(%cst_0, %12) : f64, tensor<2x2x14xf64> -> tensor<2x2x14xf64>
    %state_saved = memref.alloca() : memref<2x2x14xf64>
    // %14#2 is the final gradient of extra params
    %14:4 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %7, %arg6 = %9, %arg7 = %11, %arg8 = %13) -> (tensor<2x2x4x14xf64>, tensor<4x14xf64>, tensor<3x14xf64>, tensor<2x2x14xf64>) {
      %15 = arith.subi %c2, %arg4 : index
      %16 = memref.subview %0[%15, 0, 0, 0] [1, 2, 2, 14] [1, 1, 1, 1] : memref<3x2x2x14xf64> to memref<2x2x14xf64, #map2>
      %17 = memref.cast %16 : memref<2x2x14xf64, #map2> to memref<2x2x14xf64>
      linalg.copy(%16, %state_saved) : memref<2x2x14xf64, #map2>, memref<2x2x14xf64>
      // %18 is the cached state from the primal.
      // This is correct at this point but gets modified in the following loop.
      %18 = memref.tensor_load %17 : memref<2x2x14xf64>
      %x_original = tensor.extract_slice %arg3[%15, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
      %20 = tensor.extract_slice %arg1[0, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
      %21 = arith.mulf %x_original, %20 : tensor<14xf64>
      %22 = memref.alloc() : memref<2x14xf64>
      %23 = memref.alloc() : memref<2x2x2x14xf64>
      %24:2 = scf.for %arg9 = %c0 to %c2 step %c1 iter_args(%arg10 = %21, %arg11 = %18) -> (tensor<14xf64>, tensor<2x2x14xf64>) {
        %61 = memref.subview %22[%arg9, 0] [1, 14] [1, 1] : memref<2x14xf64> to memref<14xf64, #map3>
        %62 = memref.buffer_cast %arg10 : memref<14xf64>
        linalg.copy(%62, %61) : memref<14xf64>, memref<14xf64, #map3> 
        %63 = memref.subview %23[%arg9, 0, 0, 0] [1, 2, 2, 14] [1, 1, 1, 1] : memref<2x2x2x14xf64> to memref<2x2x14xf64, #map2>
        %64 = memref.buffer_cast %arg11 : memref<2x2x14xf64>
        linalg.copy(%64, %63) : memref<2x2x14xf64>, memref<2x2x14xf64, #map2> 
        %65 = tensor.extract_slice %arg0[%arg9, 0, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %66 = tensor.extract_slice %arg0[%arg9, 1, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %67 = tensor.extract_slice %arg0[%arg9, 0, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %68 = tensor.extract_slice %arg0[%arg9, 1, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %69 = tensor.extract_slice %arg0[%arg9, 0, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %70 = tensor.extract_slice %arg0[%arg9, 1, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %71 = tensor.extract_slice %arg0[%arg9, 0, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %72 = tensor.extract_slice %arg0[%arg9, 1, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %73 = tensor.extract_slice %arg11[%arg9, 0, 0] [1, 1, 14] [1, 1, 1] : tensor<2x2x14xf64> to tensor<14xf64>
        %74 = tensor.extract_slice %arg11[%arg9, 1, 0] [1, 1, 14] [1, 1, 1] : tensor<2x2x14xf64> to tensor<14xf64>
        %75 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg10, %65, %66 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg12: f64, %arg13: f64, %arg14: f64, %arg15: f64):  // no predecessors
          %83 = arith.mulf %arg12, %arg13 : f64
          %84 = arith.addf %83, %arg14 : f64
          %85 = call @msigmoid(%84) : (f64) -> f64
          linalg.yield %85 : f64
        } -> tensor<14xf64>
        %76 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%73, %67, %68 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg12: f64, %arg13: f64, %arg14: f64, %arg15: f64):  // no predecessors
          %83 = arith.mulf %arg12, %arg13 : f64
          %84 = arith.addf %83, %arg14 : f64
          %85 = call @msigmoid(%84) : (f64) -> f64
          linalg.yield %85 : f64
        } -> tensor<14xf64>
        %77 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg10, %69, %70 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg12: f64, %arg13: f64, %arg14: f64, %arg15: f64):  // no predecessors
          %83 = arith.mulf %arg12, %arg13 : f64
          %84 = arith.addf %83, %arg14 : f64
          %85 = call @msigmoid(%84) : (f64) -> f64
          linalg.yield %85 : f64
        } -> tensor<14xf64>
        %78 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%73, %71, %72 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg12: f64, %arg13: f64, %arg14: f64, %arg15: f64):  // no predecessors
          %83 = arith.mulf %arg12, %arg13 : f64
          %84 = arith.addf %83, %arg14 : f64
          %85 = math.tanh %84 : f64
          linalg.yield %85 : f64
        } -> tensor<14xf64>
        %79 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%74, %75, %76, %78 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg12: f64, %arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64):  // no predecessors
          %83 = arith.mulf %arg12, %arg13 : f64
          %84 = arith.mulf %arg14, %arg15 : f64
          %85 = arith.addf %83, %84 : f64
          linalg.yield %85 : f64
        } -> tensor<14xf64>
        %80 = tensor.insert_slice %79 into %arg11[%arg9, 1, 0] [1, 1, 14] [1, 1, 1] : tensor<14xf64> into tensor<2x2x14xf64>
        %81 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%77, %79 : tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg12: f64, %arg13: f64, %arg14: f64):  // no predecessors
          %83 = math.tanh %arg13 : f64
          %84 = arith.mulf %arg12, %83 : f64
          linalg.yield %84 : f64
        } -> tensor<14xf64>
        %82 = tensor.insert_slice %81 into %80[%arg9, 0, 0] [1, 1, 14] [1, 1, 1] : tensor<14xf64> into tensor<2x2x14xf64>
        scf.yield %81, %82 : tensor<14xf64>, tensor<2x2x14xf64>
      }
      %25 = tensor.extract_slice %arg1[1, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
      %26 = tensor.extract_slice %arg1[2, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
      %27 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%24#0, %25, %26 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64):  // no predecessors
        %61 = arith.mulf %arg9, %arg10 : f64
        %62 = arith.addf %61, %arg11 : f64
        linalg.yield %62 : f64
      } -> tensor<14xf64>
      %28 = call @mlogsumexp(%27) : (tensor<14xf64>) -> f64
      %29 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%27 : tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):  // no predecessors
        %61 = arith.subf %arg9, %28 : f64
        linalg.yield %61 : f64
      } -> tensor<14xf64>
      %30 = arith.subi %c3, %arg4 : index
      %31 = tensor.extract_slice %arg3[%30, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
      %32 = linalg.init_tensor [] : tensor<f64>
      %33 = linalg.fill(%cst_0, %32) : f64, tensor<f64> -> tensor<f64> 
      %34 = tensor.insert %5 into %33[] : tensor<f64>
      %35 = linalg.generic {doc = "Copy and scalar multiplication", indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel"], library_call = "sdot_grad_first"} ins(%34, %29 : tensor<f64>, tensor<14xf64>) outs(%31 : tensor<14xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
        %61 = arith.mulf %arg9, %arg10 : f64
        linalg.yield %61 : f64
      } -> tensor<14xf64>
      %36 = linalg.generic {doc = "Copy and scalar multiplication", indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel"], library_call = "sdot_grad_second"} ins(%34, %31 : tensor<f64>, tensor<14xf64>) outs(%29 : tensor<14xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
        %61 = arith.mulf %arg9, %arg10 : f64
        linalg.yield %61 : f64
      } -> tensor<14xf64>
      %37 = tensor.extract_slice %arg6[%30, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
      %38 = arith.addf %37, %35 : tensor<14xf64>
      %39 = tensor.insert_slice %38 into %arg6[%30, 0] [1, 14] [1, 1] : tensor<14xf64> into tensor<4x14xf64>
      %40 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel"]} ins(%27, %36 : tensor<14xf64>, tensor<14xf64>) outs(%cst_3 : tensor<f64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64):  // no predecessors
        %61 = arith.negf %arg10 : f64
        %62 = arith.addf %61, %arg11 : f64
        linalg.yield %62 : f64
      } -> tensor<f64>
      %41 = tensor.extract %40[] : tensor<f64>
      // This must be the computation of dypred. The grad_logsumexp matches (but the sign is reversed)
      %42 = call @__grad_mlogsumexp_arg0(%27, %41) : (tensor<14xf64>, f64) -> tensor<14xf64>
      %dypred = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%42 : tensor<14xf64>) outs(%36 : tensor<14xf64>) {
      ^bb0(%arg9: f64, %arg10: f64):  // no predecessors
        %61 = arith.addf %arg9, %arg10 : f64
        linalg.yield %61 : f64
      } -> tensor<14xf64>

      // %U = tensor.cast %dypred : tensor<14xf64> to tensor<*xf64>
      // call @print_memref_f64(%U) : (tensor<*xf64>) -> ()

      %dx_init = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%24#0, %25, %26, %dypred : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64):  // no predecessors
        %61 = arith.mulf %arg12, %arg10 : f64
        linalg.yield %61 : f64
      } -> tensor<14xf64>

      %x_times_dypred = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%24#0, %25, %26, %dypred : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
      ^bb0(%arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64):  // no predecessors
        %61 = arith.mulf %arg12, %arg9 : f64
        linalg.yield %61 : f64
      } -> tensor<14xf64>
      // dextra_params added here, then becomes %48
      // This corresponds to the dw2[2] = g line in the PyTorch reference
      %46 = tensor.extract_slice %arg7[2, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
      %47 = arith.addf %46, %dypred : tensor<14xf64>
      %48 = tensor.insert_slice %47 into %arg7[2, 0] [1, 14] [1, 1] : tensor<14xf64> into tensor<3x14xf64>

      // This must correspond to the x * g line
      %49 = tensor.extract_slice %48[1, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
      %50 = arith.addf %49, %x_times_dypred : tensor<14xf64>
      %dextra_v2 = tensor.insert_slice %50 into %48[1, 0] [1, 14] [1, 1] : tensor<14xf64> into tensor<3x14xf64>
      // The first iter arg here should be dx
      %52:3 = scf.for %arg9 = %c0 to %c2 step %c1 iter_args(%arg10 = %dx_init, %arg11 = %arg5, %arg12 = %arg8) -> (tensor<14xf64>, tensor<2x2x4x14xf64>, tensor<2x2x14xf64>) {
        %61 = arith.subi %c1, %arg9 : index
        %62 = memref.subview %22[%61, 0] [1, 14] [1, 1] : memref<2x14xf64> to memref<14xf64, #map3>
        %63 = memref.cast %62 : memref<14xf64, #map3> to memref<14xf64>
        // This is correct
        %orig_x = memref.tensor_load %63 : memref<14xf64>

        // The second time around dx is wrong
        // %U = tensor.cast %arg10: tensor<14xf64> to tensor<*xf64>
        // call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
        // This is the cached state
        // %65 = memref.subview %23[%61, 0, 0, 0] [1, 2, 2, 14] [1, 1, 1, 1] : memref<2x2x2x14xf64> to memref<2x2x14xf64, #map2>
        // %66 = memref.cast %65 : memref<2x2x14xf64, #map2> to memref<2x2x14xf64>
        // %67 = memref.tensor_load %66 : memref<2x2x14xf64>

        // This is a workaround: we're saving a copy of the state at each point because
        // otherwise our program is mutating it.
        %state_t_saved = memref.tensor_load %state_saved : memref<2x2x14xf64>
        %68 = tensor.extract_slice %arg0[%61, 0, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %69 = tensor.extract_slice %arg0[%61, 1, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %70 = tensor.extract_slice %arg0[%61, 0, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %71 = tensor.extract_slice %arg0[%61, 1, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %72 = tensor.extract_slice %arg0[%61, 0, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %73 = tensor.extract_slice %arg0[%61, 1, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %74 = tensor.extract_slice %arg0[%61, 0, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %75 = tensor.extract_slice %arg0[%61, 1, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %76 = tensor.extract_slice %state_t_saved[%61, 0, 0] [1, 1, 14] [1, 1, 1] : tensor<2x2x14xf64> to tensor<14xf64>
        %77 = tensor.extract_slice %state_t_saved[%61, 1, 0] [1, 1, 14] [1, 1, 1] : tensor<2x2x14xf64> to tensor<14xf64>
        %78 = linalg.generic {doc = "primal forget gates", indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%orig_x, %68, %69 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = call @msigmoid(%134) : (f64) -> f64
          linalg.yield %135 : f64
        } -> tensor<14xf64>
        %79 = linalg.generic {doc = "primal ingate", indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%76, %70, %71 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = call @msigmoid(%134) : (f64) -> f64
          linalg.yield %135 : f64
        } -> tensor<14xf64>
        %outgate = linalg.generic {doc = "primal outgate", indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%orig_x, %72, %73 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = call @msigmoid(%134) : (f64) -> f64
          linalg.yield %135 : f64
        } -> tensor<14xf64>
        %81 = linalg.generic {doc = "primal change", indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%76, %74, %75 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = math.tanh %134 : f64
          linalg.yield %135 : f64
        } -> tensor<14xf64>
        %cell_new = linalg.generic {doc = "primal cell", indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%77, %78, %79, %81 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.mulf %arg15, %arg16 : f64
          %135 = arith.addf %133, %134 : f64
          linalg.yield %135 : f64
        } -> tensor<14xf64>
        // %arg10 is dx
        // I think this is g * tanh(cell)
        %doutgate = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%outgate, %cell_new, %arg10 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64):  // no predecessors
          %133 = math.tanh %arg14 : f64
          %134 = arith.mulf %arg15, %133 : f64
          linalg.yield %134 : f64
        } -> tensor<14xf64>
        %84 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%outgate, %cell_new, %arg10 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64):  // no predecessors
          %133 = arith.mulf %arg15, %arg13 : f64
          %134 = math.exp %arg14 : f64
          %135 = arith.negf %arg14 : f64
          %136 = math.exp %135 : f64
          %137 = arith.addf %134, %136 : f64
          %138 = arith.mulf %137, %cst : f64
          %139 = arith.mulf %138, %138 : f64
          %140 = arith.divf %133, %139 : f64
          linalg.yield %140 : f64
        } -> tensor<14xf64>
        %85 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%77, %78, %79, %81, %84 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64, %arg18: f64):  // no predecessors
          %133 = arith.mulf %arg17, %arg14 : f64
          linalg.yield %133 : f64
        } -> tensor<14xf64>
        %86 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%77, %78, %79, %81, %84 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64, %arg18: f64):  // no predecessors
          %133 = arith.mulf %arg17, %arg13 : f64
          linalg.yield %133 : f64
        } -> tensor<14xf64>
        %87 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%77, %78, %79, %81, %84 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64, %arg18: f64):  // no predecessors
          %133 = arith.mulf %arg17, %arg16 : f64
          linalg.yield %133 : f64
        } -> tensor<14xf64>
        %88 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%77, %78, %79, %81, %84 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64, %arg18: f64):  // no predecessors
          %133 = arith.mulf %arg17, %arg15 : f64
          linalg.yield %133 : f64
        } -> tensor<14xf64>
        %89 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%76, %74, %75, %88 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = math.exp %134 : f64
          %136 = arith.negf %134 : f64
          %137 = math.exp %136 : f64
          %138 = arith.addf %135, %137 : f64
          %139 = arith.mulf %138, %cst : f64
          %140 = arith.mulf %139, %139 : f64
          %141 = arith.divf %arg16, %140 : f64
          %142 = arith.mulf %141, %arg14 : f64
          linalg.yield %142 : f64
        } -> tensor<14xf64>
        %90 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%76, %74, %75, %88 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = math.exp %134 : f64
          %136 = arith.negf %134 : f64
          %137 = math.exp %136 : f64
          %138 = arith.addf %135, %137 : f64
          %139 = arith.mulf %138, %cst : f64
          %140 = arith.mulf %139, %139 : f64
          %141 = arith.divf %arg16, %140 : f64
          %142 = arith.mulf %141, %arg13 : f64
          linalg.yield %142 : f64
        } -> tensor<14xf64>
        %91 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%76, %74, %75, %88 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = math.exp %134 : f64
          %136 = arith.negf %134 : f64
          %137 = math.exp %136 : f64
          %138 = arith.addf %135, %137 : f64
          %139 = arith.mulf %138, %cst : f64
          %140 = arith.mulf %139, %139 : f64
          %141 = arith.divf %arg16, %140 : f64
          linalg.yield %141 : f64
        } -> tensor<14xf64>
        %92 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%orig_x, %72, %73, %doutgate : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = call @msigmoid(%134) : (f64) -> f64
          %136 = call @__grad_msigmoid_arg0(%134, %arg16) : (f64, f64) -> f64
          %137 = arith.mulf %136, %arg14 : f64
          linalg.yield %137 : f64
        } -> tensor<14xf64>
        %93 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%orig_x, %72, %73, %doutgate : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = call @msigmoid(%134) : (f64) -> f64
          %136 = call @__grad_msigmoid_arg0(%134, %arg16) : (f64, f64) -> f64
          %137 = arith.mulf %136, %arg13 : f64
          linalg.yield %137 : f64
        } -> tensor<14xf64>
        %94 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%orig_x, %72, %73, %doutgate : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = call @msigmoid(%134) : (f64) -> f64
          %136 = call @__grad_msigmoid_arg0(%134, %arg16) : (f64, f64) -> f64
          linalg.yield %136 : f64
        } -> tensor<14xf64>
        %95 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%76, %70, %71, %87 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = call @msigmoid(%134) : (f64) -> f64
          %136 = call @__grad_msigmoid_arg0(%134, %arg16) : (f64, f64) -> f64
          %137 = arith.mulf %136, %arg14 : f64
          linalg.yield %137 : f64
        } -> tensor<14xf64>
        %96 = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%95 : tensor<14xf64>) outs(%89 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64):  // no predecessors
          %133 = arith.addf %arg13, %arg14 : f64
          linalg.yield %133 : f64
        } -> tensor<14xf64>
        %97 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%76, %70, %71, %87 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = call @msigmoid(%134) : (f64) -> f64
          %136 = call @__grad_msigmoid_arg0(%134, %arg16) : (f64, f64) -> f64
          %137 = arith.mulf %136, %arg13 : f64
          linalg.yield %137 : f64
        } -> tensor<14xf64>
        %98 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%76, %70, %71, %87 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = call @msigmoid(%134) : (f64) -> f64
          %136 = call @__grad_msigmoid_arg0(%134, %arg16) : (f64, f64) -> f64
          linalg.yield %136 : f64
        } -> tensor<14xf64>
        %99 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%orig_x, %68, %69, %86 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = call @msigmoid(%134) : (f64) -> f64
          %136 = call @__grad_msigmoid_arg0(%134, %arg16) : (f64, f64) -> f64
          %137 = arith.mulf %136, %arg14 : f64
          linalg.yield %137 : f64
        } -> tensor<14xf64>
        %dx_next = linalg.generic {doc = "Add in place", indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%99 : tensor<14xf64>) outs(%92 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64):  // no predecessors
          %133 = arith.addf %arg13, %arg14 : f64
          linalg.yield %133 : f64
        } -> tensor<14xf64>
        %101 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%orig_x, %68, %69, %86 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = call @msigmoid(%134) : (f64) -> f64
          %136 = call @__grad_msigmoid_arg0(%134, %arg16) : (f64, f64) -> f64
          %137 = arith.mulf %136, %arg13 : f64
          linalg.yield %137 : f64
        } -> tensor<14xf64>
        %102 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%orig_x, %68, %69, %86 : tensor<14xf64>, tensor<14xf64>, tensor<14xf64>, tensor<14xf64>) outs(%cst_2 : tensor<14xf64>) {
        ^bb0(%arg13: f64, %arg14: f64, %arg15: f64, %arg16: f64, %arg17: f64):  // no predecessors
          %133 = arith.mulf %arg13, %arg14 : f64
          %134 = arith.addf %133, %arg15 : f64
          %135 = call @msigmoid(%134) : (f64) -> f64
          %136 = call @__grad_msigmoid_arg0(%134, %arg16) : (f64, f64) -> f64
          linalg.yield %136 : f64
        } -> tensor<14xf64>
        // updates to dstate happen here.
        %103 = tensor.extract_slice %arg12[%61, 1, 0] [1, 1, 14] [1, 1, 1] : tensor<2x2x14xf64> to tensor<14xf64>
        %104 = arith.addf %103, %85 : tensor<14xf64>
        %105 = tensor.insert_slice %104 into %arg12[%61, 1, 0] [1, 1, 14] [1, 1, 1] : tensor<14xf64> into tensor<2x2x14xf64>
        %106 = tensor.extract_slice %105[%61, 0, 0] [1, 1, 14] [1, 1, 1] : tensor<2x2x14xf64> to tensor<14xf64>
        %107 = arith.addf %106, %96 : tensor<14xf64>
        %dstate_next = tensor.insert_slice %107 into %105[%61, 0, 0] [1, 1, 14] [1, 1, 1] : tensor<14xf64> into tensor<2x2x14xf64>

        %109 = tensor.extract_slice %arg11[%61, 1, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %110 = arith.addf %109, %91 : tensor<14xf64>
        %111 = tensor.insert_slice %110 into %arg11[%61, 1, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<14xf64> into tensor<2x2x4x14xf64>
        %112 = tensor.extract_slice %111[%61, 0, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %113 = arith.addf %112, %90 : tensor<14xf64>
        %114 = tensor.insert_slice %113 into %111[%61, 0, 3, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<14xf64> into tensor<2x2x4x14xf64>
        %115 = tensor.extract_slice %114[%61, 1, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %116 = arith.addf %115, %94 : tensor<14xf64>
        %117 = tensor.insert_slice %116 into %114[%61, 1, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<14xf64> into tensor<2x2x4x14xf64>
        %118 = tensor.extract_slice %117[%61, 0, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %119 = arith.addf %118, %93 : tensor<14xf64>
        %120 = tensor.insert_slice %119 into %117[%61, 0, 2, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<14xf64> into tensor<2x2x4x14xf64>
        %121 = tensor.extract_slice %120[%61, 1, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %122 = arith.addf %121, %98 : tensor<14xf64>
        %123 = tensor.insert_slice %122 into %120[%61, 1, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<14xf64> into tensor<2x2x4x14xf64>
        %124 = tensor.extract_slice %123[%61, 0, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %125 = arith.addf %124, %97 : tensor<14xf64>
        %126 = tensor.insert_slice %125 into %123[%61, 0, 1, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<14xf64> into tensor<2x2x4x14xf64>
        %127 = tensor.extract_slice %126[%61, 1, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %128 = arith.addf %127, %102 : tensor<14xf64>
        %129 = tensor.insert_slice %128 into %126[%61, 1, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<14xf64> into tensor<2x2x4x14xf64>
        %130 = tensor.extract_slice %129[%61, 0, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<2x2x4x14xf64> to tensor<14xf64>
        %131 = arith.addf %130, %101 : tensor<14xf64>
        %132 = tensor.insert_slice %131 into %129[%61, 0, 0, 0] [1, 1, 1, 14] [1, 1, 1, 1] : tensor<14xf64> into tensor<2x2x4x14xf64>
        scf.yield %dx_next, %132, %dstate_next : tensor<14xf64>, tensor<2x2x4x14xf64>, tensor<2x2x14xf64>
      } // end of adjoint predict loop
      %53 = arith.mulf %52#0, %20 : tensor<14xf64>
      // x original matches, meaning the problem must be %52#0 (which should be dx)
      %54 = arith.mulf %52#0, %x_original : tensor<14xf64>
      // %U = tensor.cast %52#0 : tensor<14xf64> to tensor<*xf64>
      // call @print_memref_f64(%U) : (tensor<*xf64>) -> ()

      // This is the final place where dextra is different.
      %55 = tensor.extract_slice %dextra_v2[0, 0] [1, 14] [1, 1] : tensor<3x14xf64> to tensor<14xf64>
      %56 = arith.addf %55, %54 : tensor<14xf64>
      %dextra_next = tensor.insert_slice %56 into %dextra_v2[0, 0] [1, 14] [1, 1] : tensor<14xf64> into tensor<3x14xf64>
      %58 = tensor.extract_slice %39[%15, 0] [1, 14] [1, 1] : tensor<4x14xf64> to tensor<14xf64>
      %59 = arith.addf %58, %53 : tensor<14xf64>
      %60 = tensor.insert_slice %59 into %39[%15, 0] [1, 14] [1, 1] : tensor<14xf64> into tensor<4x14xf64>
      scf.yield %52#1, %60, %dextra_next, %52#2 : tensor<2x2x4x14xf64>, tensor<4x14xf64>, tensor<3x14xf64>, tensor<2x2x14xf64>
    }
    return %14#0, %14#2 : tensor<2x2x4x14xf64>, tensor<3x14xf64>
  }
  func private @print_memref_f64(tensor<*xf64>) attributes {llvm.emit_c_interface}
  func @main() {
    %cst = arith.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<4x14xf64>
    %cst_0 = arith.constant dense<[[[0.93681999999999998, 4.857800e-01, 1.406500e-01, 1.632000e-01, 3.576900e-01, 2.958300e-02, 0.79632999999999998, 8.438400e-01, 3.312100e-01, 5.709600e-01, 6.965400e-01, 6.607800e-01, 9.545500e-01, 4.287600e-01], [2.570500e-01, 7.761700e-01, 8.659300e-01, 6.295800e-01, 6.512800e-01, 7.005400e-01, 1.829400e-01, 4.810300e-02, 3.534300e-01, 1.712900e-01, 8.219600e-01, 2.243600e-01, 3.864200e-01, 6.748200e-01]], [[1.886600e-01, 3.644500e-01, 1.834900e-01, 0.94308999999999998, 1.414000e-01, 1.486100e-01, 7.793700e-01, 2.243600e-02, 0.90437999999999996, 7.897200e-01, 9.244200e-01, 4.566300e-01, 2.287400e-02, 4.415100e-01], [5.608000e-01, 9.927700e-01, 2.062400e-01, 4.172600e-01, 4.430700e-01, 7.398400e-01, 1.845700e-01, 1.801600e-01, 1.816200e-01, 4.530900e-01, 3.025700e-01, 5.744000e-02, 4.362300e-01, 2.600300e-01]]]> : tensor<2x2x14xf64>
    %cst_1 = arith.constant dense<[[5.848500e-01, 4.108000e-01, 1.904900e-01, 2.852500e-01, 1.910400e-01, 6.738100e-01, 1.841500e-03, 1.619900e-01, 1.181500e-01, 2.421000e-01, 1.894400e-01, 1.250800e-01, 5.332500e-01, 2.907200e-01], [5.857700e-01, 4.042700e-01, 7.089400e-01, 7.267200e-01, 4.806000e-01, 6.333100e-01, 4.844800e-01, 4.773300e-01, 9.411800e-01, 5.795300e-01, 4.686000e-01, 7.239600e-02, 1.457200e-01, 1.783400e-01], [7.438200e-01, 6.332500e-01, 7.420500e-02, 2.270400e-02, 8.045800e-01, 6.451900e-02, 5.636300e-01, 7.936300e-01, 2.665900e-01, 8.200000e-01, 5.087000e-01, 6.127400e-01, 5.137800e-01, 1.167100e-01]]> : tensor<3x14xf64>
    %cst_2 = arith.constant dense<"0x04392861A6EDE73F376C5B94D920D13FC7F484251E50E33FAFCE31207BBDEC3F8E01D9EBDD1FE23F2C4833164D67E63F2E39EE940ED6EA3FCBB91457957DD13FBCB376DB85E6CA3FA857CA32C4B1CE3FF584251E5036E83F0C7558E1968FB83F732EC55565DFEA3FDBC4C9FD0E45C93F4EB9C2BB5CC4D73FBF2B82FFAD64D53F732EC55565DFE63F17B7D100DE02E63F8B6CE7FBA9F1D83F1FBFB7E9CF7EB83F6E3480B74082E93F1618B2BAD573EF3FAAF1D24D6210E03F78978BF84ECCCA3FDA8F14916115BF3FE50AEF7211DFE03F3C4ED1915CFEEC3F8109DCBA9BA7BA3F2B8716D9CEF7E63FB876A22424D2B23F156F641EF983E63F6A87BF266BD4E23F9BAC510FD1E8C23F3BAA9A20EA3ED63FCFF753E3A59BC43F68D0D03FC1C5D23FC748F60835439A3F18CDCAF6216FB93FAC1C5A643BDFDF3FFB57569A9482E53F19C00067CE9F7B3F00AE64C74620C63FC1A8A44E4013E73FF853E3A59BC4EC3F3ED00A0C59DDDE3F560E2DB29DEFE93FB936548CF337EF3F51BD35B05582E13F17D4B7CCE9B2E93F7F4DD6A88768E73FE8BCC62E51BDD73F5FD218ADA3AAC93F7EE36BCF2C09D23FC3BB5CC47762E03FD5B2B5BE4868E63F25404D2D5BEBE63F5114E8137992D03FDE72F56393FCB83F912749D74CBEC53F7C0F971C774AE03F5FEFFE78AF5AD33FC2FA3F87F9F2EB3F29E8F692C668D73FC2120F289B72D53FA0C37C7901F6E43FCAC342AD69DED33F350708E6E8F1E53FBA1457957D57E93F799274CDE49BDF3F01A4367172BFED3F481630815B77EA3FBAF770C971A7D23F6BD44334BA83D63FB1A206D3307CE23F042159C0046EE23F28D53E1D8F19E13F745E6397A8DEED3F57EC2FBB270FE03F17F19D98F562D63FF2D24D621058E53FFD135CACA8C1CC3F34D769A4A5F2D63F742497FF907EC33F836E2F698CD6E93F179AEB34D252CD3F224F92AE997CE93F9D8026C286A7D53FA27F828B1535E63FF8C264AA6054ED3F4030478FDFDB943FDDEEE53E390AA83FF9BD4D7FF623E13FAF997CB3CD8DE03F78D498107349AD3F174850FC1873E33F2D431CEBE236E53F252367614F3BE33F3F1D8F19A88CE13FCA32C4B12E6ECB3F29B16B7BBB25893F8A1EF818AC38A53F5917B7D100DED03FC746205ED72FEA3F73637AC2120FD83F598B4F01309EE93F300DC347C494E13FF9A067B3EA73E23F4B3CA06CCA15C23F825660C8EA56E73FAB21718FA50FEE3FCB10C7BAB88DE83F41481630815BE93F9F8EC70C54C6ED3F33F9669B1BD3E33F8E9257E71890E13FEC17EC866D8BE53FFFE7305F5E80CD3F0B410E4A9869D13F1E166A4DF38EEB3F618E1EBFB7E9E23F1361C3D32B65E23F62BEBC00FBE8D03F13F241CF66D5E43F003ACC971760DB3FF1BA7EC16ED8EA3F33A8363811FDB23F66BD18CA8976ED3F2844C02154A9E03FE88711C2A38DEF3FCC5D4BC8073DEA3F315F5E807D74D43FAF997CB3CD8DE93F1CF0F96184F0E73F0742B28009DCE13F02BC0512143FE93FFFE7305F5E80EE3F787AA52C431CE23F0AF4893C49BAE63F1973D712F241E43F3C122F4FE78A623F2575029A081BE53F50FC1873D712E33FD93D7958A835E73F6536C8242367C53F3C8386FE092ED43FD4D4B2B5BE48E33F44696FF085C9BC3FA7AE7C96E7C1D53F4F5DF92CCF83D53F7FBC57AD4CF8E73F59FAD005F52DEA3FF7E978CC4065E53FDCBA9BA73AE4C23F16FBCBEEC9C3BA3F1349F4328AE5C63F1990BDDEFDF1E13F4D840D4FAF94E23FBE4BA94BC631923F632827DA5548C93F079964E42CECEB3F90D8EE1EA0FB923F2FFA0AD28C45C73F7C0F971C774AD73F7380608E1EBFED3FD717096D3997DE3FF8AA9509BFD4E33F6F8104C58F31EC3F8C2D04392861DA3F53793BC269C1CB3F39807EDFBF79B93F37548CF337A1DC3FA60F5D50DF32CB3FB875374F75C8E93F8C101E6D1CB1D63F98512CB7B41AE43F9A7CB3CD8DE9EA3F72DC291DACFFD93FE2E47E87A240E43FD9CEF753E3A5EF3FFCC6D79E5912E93FEC12D55B035BE03FEA094B3CA06CBA3FF19D98F56228EE3F9CA73AE466B8E23F2592E86514CBD13FDC9DB5DB2E34D73FCCECF318E599B73F293FA9F6E978EC3F3F6EBF7CB262A83F90BC73284355B83FF5B9DA8AFD65DD3FA6ED5F596952CA3F4D327216F6B4EE3FC2DD59BBED42EE3F31B1F9B83654C03F6EA301BC0512EA3F143FC6DCB584C43F3E7958A835CDE13FE55FCB2BD7DBA63F2D2460747973B83FF71E2E39EE94E83F2C2B4D4A41B7CF3F30BB270F0BB5EF3FA0C37C7901F6D93FC763062AE3DFD53F68B3EA73B515EB3FC63368E89FE0DA3F0A80F10C1AFAE33FD15CA79196CADB3FD5264EEE7728E13FC173EFE192E3E43F9EEFA7C64B37EC3F5E807D74EACADD3F08E6E8F17B9BC63FAF25E4839ECDEB3F51F701486DE2EB3F0395F1EF332EBC3F21CD58349D9DE43F0074982F2FC0DC3F8F8AFF3BA242A53FB8CCE9B298D8EA3FF3716DA818E7E93FFCFB8C0B0742DC3FB459F5B9DA8AEE3F"> : tensor<2x2x4x14xf64>
    %0:2 = call @__grad_mlstm_objective(%cst_2, %cst_1, %cst_0, %cst) : (tensor<2x2x4x14xf64>, tensor<3x14xf64>, tensor<2x2x14xf64>, tensor<4x14xf64>) -> (tensor<2x2x4x14xf64>, tensor<3x14xf64>)
    %1 = tensor.cast %0#1 : tensor<3x14xf64> to tensor<*xf64>
    // call @print_memref_f64(%1) : (tensor<*xf64>) -> ()
    return
  }
}

