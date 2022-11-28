#map8 = affine_map<(d0, d1) -> (d0, d1)>
#map9 = affine_map<(d0, d1) -> (d0)>

// first arg: dense, (d0, d1) -> (d0)
// secnd arg: sparse, dense, (d0, d1) -> (d0, d1)
// output : (d0, d1) -> (d0, d1)

// d0 is a parallel iterator along a sparse dim
// we need one loop, d1, across the dense dimension
// ivs: sparse_index, d1

// produces sparse x dense, the indexing map is also (d0, d1) -> (d0, d1)
func @rowhot_broadcast_mul(%39: tensor<12xf64>, %16: tensor<12x3xf64, "rowhot">) -> tensor<12x3xf64> {
  %cst_3 = arith.constant dense<0.0> : tensor<12x3xf64>
  %40 = linalg.generic
    {indexing_maps = [#map9, #map8, #map8], iterator_types = ["parallel", "parallel"]}
    ins(%39, %16 : tensor<12xf64>, tensor<12x3xf64, "rowhot">)
    outs(%cst_3 : tensor<12x3xf64>) {
  ^bb0(%arg11: f64, %arg12: f64, %arg13: f64):  // no predecessors
    %47 = arith.mulf %arg12, %arg11 : f64
    %48 = arith.addf %47, %arg13 : f64
    linalg.yield %48 : f64
  } -> tensor<12x3xf64>
  return %40 : tensor<12x3xf64>
}
