#elementwise_2d = affine_map<(d0, d1) -> (d0, d1)>
#first_2d = affine_map<(d0, d1) -> (d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map9 = affine_map<(d0, d1) -> (d1)>
#map10 = affine_map<(d0) -> (d0)>
#map11 = affine_map<(d0) -> ()>
#map12 = affine_map<(d0, d1, d2) -> (d0)>
func @mlir_elementwise_exp(%arg0: tensor<{{k}}x{{d}}xf64>) -> tensor<{{k}}x{{d}}xf64> {
  // %0 = constant dense<0.0> : tensor<{{k}}x{{d}}xf64>
  // %1 = linalg.generic { indexing_maps = [#elementwise_2d, #elementwise_2d], iterator_types = ["parallel", "parallel"] } ins(%arg0 : tensor<{{k}}x{{d}}xf64>) outs(%0 : tensor<{{k}}x{{d}}xf64>) {
  // ^bb0(%arg1: f64, %arg2: f64):
  //   %2 = math.exp %arg1 : f64
  //   linalg.yield %2 : f64
  // } -> tensor<{{k}}x{{d}}xf64>
  %0 = math.exp %arg0 : tensor<{{k}}x{{d}}xf64>
  return %0 : tensor<{{k}}x{{d}}xf64>
}

func @mlir_row_sum(%arg0: tensor<{{k}}x{{d}}xf64>) -> tensor<{{k}}xf64> {
  %init = constant dense<0.0> : tensor<{{k}}xf64>
  %1 = linalg.generic
    {indexing_maps = [#elementwise_2d, #first_2d], iterator_types = ["parallel", "reduction"]}
    ins(%arg0 : tensor<{{k}}x{{d}}xf64>)
    outs(%init : tensor<{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %43 = addf %arg7, %arg8 : f64
    linalg.yield %43 : f64
  } -> tensor<{{k}}xf64>
  return %1 : tensor<{{k}}xf64>
}

func @mlir_broadcasted_sub(%arg0: tensor<{{n}}x{{d}}xf64>, %arg1: tensor<{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64> {
  %cst_7 = constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf64>
  %2 = linalg.generic
    {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<{{n}}x{{d}}xf64>, tensor<{{k}}x{{d}}xf64>)
    outs(%cst_7 : tensor<{{n}}x{{k}}x{{d}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %43 = subf %arg7, %arg8 : f64
    linalg.yield %43 : f64
  } -> tensor<{{n}}x{{k}}x{{d}}xf64>
  return %2 : tensor<{{n}}x{{k}}x{{d}}xf64>
}


func @mlir_inner_einsum(%arg0: tensor<{{k}}x{{d}}x{{d}}xf64>, %arg1: tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64> {
  %cst_7 = constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf64>
  %3 = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    }
    ins(%arg0, %arg1 : tensor<{{k}}x{{d}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>)
    outs(%cst_7 : tensor<{{n}}x{{k}}x{{d}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %43 = mulf %arg7, %arg8 : f64
    %44 = addf %43, %arg9 : f64
    linalg.yield %44 : f64
  } -> tensor<{{n}}x{{k}}x{{d}}xf64>
  return %3 : tensor<{{n}}x{{k}}x{{d}}xf64>
}


func @mlir_squared_mult(%0: tensor<{{k}}x{{d}}xf64>, %2: tensor<{{n}}x{{k}}x{{d}}xf64>, %3: tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}x{{d}}xf64> {
  %cst_7 = constant dense<0.0> : tensor<{{n}}x{{k}}x{{d}}xf64>
  %4 = linalg.generic
    {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%0, %2, %3 : tensor<{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>, tensor<{{n}}x{{k}}x{{d}}xf64>)
    outs(%cst_7 : tensor<{{n}}x{{k}}x{{d}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):  // no predecessors
    %43 = mulf %arg7, %arg8 : f64
    %44 = addf %43, %arg9 : f64
    %45 = mulf %44, %44 : f64
    linalg.yield %45 : f64
  } -> tensor<{{n}}x{{k}}x{{d}}xf64>
  return %4 : tensor<{{n}}x{{k}}x{{d}}xf64>
}

func @mlir_batched_rowsum(%arg0: tensor<{{n}}x{{k}}x{{d}}xf64>) -> tensor<{{n}}x{{k}}xf64> {
  %cst_6 = constant dense<0.0> : tensor<{{n}}x{{k}}xf64>
  %5 = linalg.generic
    {indexing_maps = [#map4, #map8], iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0 : tensor<{{n}}x{{k}}x{{d}}xf64>)
    outs(%cst_6 : tensor<{{n}}x{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64):  // no predecessors
    %43 = addf %arg7, %arg8 : f64
    linalg.yield %43 : f64
  } -> tensor<{{n}}x{{k}}xf64>
  return %5 : tensor<{{n}}x{{k}}xf64>
}

func @mlir_scalar_mult_sub(%6: tensor<{{k}}xf64>, %5: tensor<{{n}}x{{k}}xf64>) -> tensor<{{n}}x{{k}}xf64> {
  %cst_5 = constant 0.5 : f64
  %cst_6 = constant dense<0.0> : tensor<{{n}}x{{k}}xf64>
  %7 = linalg.generic
    {indexing_maps = [#map9, #map0, #map0], iterator_types = ["parallel", "parallel"]}
    ins(%6, %5 : tensor<{{k}}xf64>, tensor<{{n}}x{{k}}xf64>)
    outs(%cst_6 : tensor<{{n}}x{{k}}xf64>) {
  ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):  // no predecessors
    %43 = mulf %arg8, %cst_5 : f64
    %44 = subf %arg7, %43 : f64
    linalg.yield %44 : f64
  } -> tensor<{{n}}x{{k}}xf64>
  return %7 : tensor<{{n}}x{{k}}xf64>
}
