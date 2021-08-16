# Differentiating a `linalg.generic` op

This is the crown jewel of differentiating through linalg ops, especially because the [named ops are considered legacy](https://llvm.discourse.group/t/can-i-extend-linalg-conv-to-support-tensor-as-input-and-output/4020/4).

# Dot product

Primal computation:
```mlir
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
%0 = linalg.generic
  {indexing_maps = [#map0, #map0, #map1],
  iterator_types = ["reduction"]}
  ins(%arg0, %arg1 : tensor<131072xf32>, tensor<131072xf32>)
  outs(%cst : tensor<f32>) {
^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
  %1 = mulf %arg2, %arg3 : f32
  %2 = addf %arg4, %1 : f32
  linalg.yield %2 : f32
} -> tensor<f32>
```

Adjoint (first argument):
```mlir
%g = constant dense<1.0> : tensor<f32>
%cst = constant dense<0.0> : tensor<4xf32>
%0 = linalg.generic
  {indexing_maps = [#map1, #map0, #map0],
  iterator_types=["parallel"]}
  ins(%g, %arg1 : tensor<f32>, tensor<4xf32>)
  outs(%cst : tensor<4xf32>) {
^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
  %1 = mulf %arg2, %arg3 : f32
  %2 = addf %arg4, %1 : f32
  linalg.yield %2 : f32
} -> tensor<4xf32>
```

- Indexing maps were directly swapped along with the argument (1, 3)
- Iterator type went from reduction to parallel
- The body can be simplified technically but this may not yield significant improvements.

# Matrix-vector multiplication

Primal:
```mlir
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
%0 = linalg.generic
  {indexing_maps = [#map0, #map1, #map2],
  iterator_types = ["parallel", "reduction"]}
  ins(%arg0, %arg1 : tensor<512x1024xf32>, tensor<1024xf32>)
  outs(%cst : tensor<512xf32>) {
^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
  %1 = mulf %arg2, %arg3 : f32
  %2 = addf %arg4, %1 : f32
  linalg.yield %2 : f32
} -> tensor<512xf32>
```

Adjoint (first argument):
```mlir
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
%0 = linalg.generic {
  doc = "Vector-vector outer product",
  indexing_maps = [#map0, #map1, #map2],
  iterator_types = ["parallel", "parallel"],
  library_call = "souter"}
  ins(%cst, %arg1 : tensor<512xf32>, tensor<1024xf32>)
  outs(%arg0 : tensor<512x1024xf32>) {
^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
  %1 = mulf %arg2, %arg3 : f32
  linalg.yield %1 : f32
} -> tensor<512x1024xf32>
```

- First arg switched places with the output
- The `indexing_maps` also switched places
- Reduction iterator became parallel while first parallel iterator stayed parallel

# Vector-matrix multiplication

Primal:
```mlir
#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
%0 = linalg.generic {
  indexing_maps = [#map0, #map1, #map2],
  iterator_types = ["parallel", "reduction"]
  }
  ins(%arg0, %arg1 : tensor<3xf32>, tensor<3x4xf32>)
  outs(%cst : tensor<4xf32>) {
^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
  %2 = mulf %arg2, %arg3 : f32
  %3 = addf %arg4, %2 : f32
  linalg.yield %3 : f32
} -> tensor<4xf32>
```

Adjoint (first argument):
```mlir
#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
%0 = linalg.generic {
  indexing_maps = [#map0, #map1, #map2],
  iterator_types = ["parallel", "reduction"]
  }
  ins(%cst, %arg1 : tensor<4xf32>, tensor<3x4xf32>)
  outs(%cst_0 : tensor<3xf32>) {
^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
  %1 = mulf %arg2, %arg3 : f32
  %2 = addf %1, %arg4 : f32
  linalg.yield %2 : f32
} -> tensor<3xf32>
```

- `map1` had its arguments switched
- The iterators stayed the ame

Adjoint (second argument):
```mlir
#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
%0 = linalg.generic {
  indexing_maps = [#map0, #map1, #map2],
  iterator_types = ["parallel", "parallel"]}
  ins(%cst, %arg0 : tensor<4xf32>, tensor<3xf32>)
  outs(%arg1 : tensor<3x4xf32>) {
^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
  %1 = mulf %arg2, %arg3 : f32
  linalg.yield %1 : f32
} -> tensor<3x4xf32>
```

# Matrix multiplication

Primal computation:
```mlir
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

%0 = linalg.generic
  {indexing_maps = [#map0, #map1, #map2],
   iterator_types = ["parallel", "parallel", "reduction"]}
  ins(%arg0, %arg1 : tensor<3x4xf32>, tensor<4x5xf32>)
  outs(%cst : tensor<3x5xf32>) {
^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
  %1 = mulf %arg2, %arg3 : f32
  %2 = addf %arg4, %1 : f32
  linalg.yield %2 : f32
} -> tensor<3x5xf32>
```

Adjoint (first argument):
```mlir
%0 = linalg.generic
  {indexing_maps = [#map2, #map1, #map0],
   iterator_types = ["parallel", "reduction", "parallel"]}
  ins(%cst, %arg1 : tensor<3x5xf32>, tensor<4x5xf32>)
  outs(%cst_0 : tensor<3x4xf32>) {
^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
  %1 = mulf %arg2, %arg3 : f32
  %2 = addf %arg4, %1 : f32
  linalg.yield %2 : f32
} -> tensor<3x4xf32>
```

- The outermost iterators remained parallel while the other two loops switched
- The indexing_maps got rearranged

# Differentiation strategies

- Can I infer the maps directly, perhaps?
- Looks like switching the maps and just using parallel everything is working
almost a little too well. It works for dot products, matvec, and vecmat. Maybe even matmul?
