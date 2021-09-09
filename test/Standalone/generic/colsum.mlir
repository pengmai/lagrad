func @colsum(%arg0 : tensor<4x5xf32>) -> tensor<5xf32> {
  %cst = constant dense<0.0> : tensor<5xf32>
  %0 = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]
    }
    ins(%arg0 : tensor<4x5xf32>)
    outs(%cst : tensor<5xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = addf %arg1, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

func @main() {
  %cst = constant dense<[
    [0.95033034, 0.04059763, 0.63056314, 0.21032931, 0.9302365 ],
    [0.46717497, 0.72913109, 0.00538126, 0.63545051, 0.46740388],
    [0.7444556 , 0.70883291, 0.27864411, 0.54437905, 0.7263821 ],
    [0.36535766, 0.55929595, 0.70579915, 0.17196087, 0.39675371]
  ]> : tensor<4x5xf32>

  %f = constant @colsum : (tensor<4x5xf32>) -> tensor<5xf32>
  %df = standalone.grad %f {of = [0]}: (tensor<4x5xf32>) -> tensor<5xf32>, (tensor<4x5xf32>) -> tensor<4x5xf32>
  %res = call_indirect %df(%cst) : (tensor<4x5xf32>) -> tensor<4x5xf32>
  %U = tensor.cast %res : tensor<4x5xf32> to tensor<*xf32>
  call @print_memref_f32(%U) : (tensor<*xf32>) -> ()
  return
}
