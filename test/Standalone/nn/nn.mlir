// #map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#relu2d = {
  doc = "ReLU 2D",
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
}

func @batched_cross_entropy(%activations: tensor<10x64xf32>, %labels: tensor<64xi32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cd = arith.constant 10 : index
  %cb = arith.constant 64 : index
  %cbf = arith.constant 64.0 : f32
  %max_init = tensor.extract_slice %activations[0, 0] [1, 64] [1, 1] : tensor<10x64xf32> to tensor<64xf32>
  %max = scf.for %iv = %c1 to %cd step %c1 iter_args(%max_outer = %max_init) -> tensor<64xf32> {
    %max_outer_next = scf.for %bv = %c0 to %cb step %c1 iter_args(%max_it = %max_outer) -> tensor<64xf32> {
      %ai = tensor.extract %activations[%iv, %bv] : tensor<10x64xf32>
      %max_val = tensor.extract %max_it[%bv] : tensor<64xf32>
      %p = arith.cmpf ogt, %ai, %max_val : f32
      %max_val_next = select %p, %ai, %max_val : f32
      %max_next = tensor.insert %max_val_next into %max_it[%bv] : tensor<64xf32>
      scf.yield %max_next : tensor<64xf32>
    }
    scf.yield %max_outer_next : tensor<64xf32>
  }

  %zerot = arith.constant dense<0.0> : tensor<10x64xf32>
  %zero2 = arith.constant dense<0.0> : tensor<64xf32>
  %exp = linalg.generic
    { indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"] }
    ins(%activations, %max : tensor<10x64xf32>, tensor<64xf32>)
    outs(%zerot : tensor<10x64xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.subf %arg0, %arg1 : f32
    %1 = math.exp %0 : f32
    linalg.yield %1 : f32
  } -> tensor<10x64xf32>

  %total = linalg.generic
    { indexing_maps = [#map1, #map2], iterator_types = ["reduction", "parallel"] }
    ins(%exp : tensor<10x64xf32>)
    outs(%zero2 : tensor<64xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<64xf32>

  %zero = arith.constant 0.0 : f32
  %cross_entropy_total = scf.for %iv = %c0 to %cb step %c1 iter_args(%ce_it = %zero) -> f32 {
    %label_i32 = tensor.extract %labels[%iv] : tensor<64xi32>
    %label = arith.index_cast %label_i32 : i32 to index
    %exp_val = tensor.extract %exp[%label, %iv] : tensor<10x64xf32>
    %total_val = tensor.extract %total[%iv] : tensor<64xf32>
    %sm_val = arith.divf %exp_val, %total_val : f32
    %ll = math.log %sm_val : f32
    %nll = arith.negf %ll : f32
    %ce_next = arith.addf %ce_it, %nll : f32
    scf.yield %ce_next : f32
  }
  %primal = arith.divf %cross_entropy_total, %cbf : f32
  return %primal : f32
}

// func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }
// func private @print_memref_i32(tensor<*xi32>) attributes { llvm.emit_c_interface }

#broadcast_add = {
  doc = "Broadcasted add",
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
}

func @mlir_mlp_batched(
  %input_t: tensor<784x64xf32>,
  %labels: tensor<64xi32>,
  %w0: tensor<512x784xf32>,
  %b0: tensor<512xf32>,
  %w1: tensor<512x512xf32>,
  %b1: tensor<512xf32>,
  %w2: tensor<10x512xf32>,
  %b2: tensor<10xf32>
) -> f32 {
  %zero = arith.constant 0.0 : f32
  %zero_0 = arith.constant dense<0.0> : tensor<512x64xf32>
  %zero_1 = arith.constant dense<0.0> : tensor<512x64xf32>
  %zero_2 = arith.constant dense<0.0> : tensor<10x64xf32>
  %h0_0 = linalg.matmul ins(%w0, %input_t : tensor<512x784xf32>, tensor<784x64xf32>) outs(%zero_0 : tensor<512x64xf32>) -> tensor<512x64xf32>
  %h0_1 = linalg.generic #broadcast_add ins(%h0_0, %b0 : tensor<512x64xf32>, tensor<512xf32>) outs(%zero_0 : tensor<512x64xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<512x64xf32>
  %h0 = linalg.generic #relu2d ins(%h0_1 : tensor<512x64xf32>) outs(%zero_0 : tensor<512x64xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %p = arith.cmpf ogt, %arg0, %zero : f32
    %0 = select %p, %arg0, %zero : f32
    linalg.yield %0 : f32
  } -> tensor<512x64xf32>

  %h1_0 = linalg.matmul ins(%w1, %h0 : tensor<512x512xf32>, tensor<512x64xf32>) outs(%zero_1 : tensor<512x64xf32>) -> tensor<512x64xf32>
  %h1_1 = linalg.generic #broadcast_add ins(%h1_0, %b1 : tensor<512x64xf32>, tensor<512xf32>) outs(%zero_1 : tensor<512x64xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<512x64xf32>
  %h1 = linalg.generic #relu2d ins(%h1_1 : tensor<512x64xf32>) outs(%zero_1 : tensor<512x64xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %p = arith.cmpf ogt, %arg0, %zero : f32
    %0 = select %p, %arg0, %zero : f32
    linalg.yield %0 : f32
  } -> tensor<512x64xf32>

  %act_0 = linalg.matmul ins(%w2, %h1 : tensor<10x512xf32>, tensor<512x64xf32>) outs(%zero_2 : tensor<10x64xf32>) -> tensor<10x64xf32>
  %act = linalg.generic #broadcast_add ins(%act_0, %b2 : tensor<10x64xf32>, tensor<10xf32>) outs(%zero_2 : tensor<10x64xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<10x64xf32>
  %loss = call @batched_cross_entropy(%act, %labels) : (tensor<10x64xf32>, tensor<64xi32>) -> f32
  return %loss : f32
}

func @lagrad_mlp_batched(
  %input_t: tensor<784x64xf32>,
  %labels: tensor<64xi32>,
  %w0: tensor<512x784xf32>,
  %b0: tensor<512xf32>,
  %w1: tensor<512x512xf32>,
  %b1: tensor<512xf32>,
  %w2: tensor<10x512xf32>,
  %b2: tensor<10xf32>
) -> (
  tensor<512x784xf32>,
  tensor<512xf32>,
  tensor<512x512xf32>,
  tensor<512xf32>,
  tensor<10x512xf32>,
  tensor<10xf32>
) {
  %f = constant @mlir_mlp_batched : (
    tensor<784x64xf32>,
    tensor<64xi32>,
    tensor<512x784xf32>,
    tensor<512xf32>,
    tensor<512x512xf32>,
    tensor<512xf32>,
    tensor<10x512xf32>,
    tensor<10xf32>
  ) -> f32
  %df = standalone.grad %f {of = [2, 3, 4, 5, 6, 7]} : (
    tensor<784x64xf32>,
    tensor<64xi32>,
    tensor<512x784xf32>,
    tensor<512xf32>,
    tensor<512x512xf32>,
    tensor<512xf32>,
    tensor<10x512xf32>,
    tensor<10xf32>
  ) -> f32, (
    tensor<784x64xf32>,
    tensor<64xi32>,
    tensor<512x784xf32>,
    tensor<512xf32>,
    tensor<512x512xf32>,
    tensor<512xf32>,
    tensor<10x512xf32>,
    tensor<10xf32>
  ) -> (
    tensor<512x784xf32>,
    tensor<512xf32>,
    tensor<512x512xf32>,
    tensor<512xf32>,
    tensor<10x512xf32>,
    tensor<10xf32>
  )
  %res:6 = call_indirect %df(%input_t, %labels, %w0, %b0, %w1, %b1, %w2, %b2) : (
    tensor<784x64xf32>,
    tensor<64xi32>,
    tensor<512x784xf32>,
    tensor<512xf32>,
    tensor<512x512xf32>,
    tensor<512xf32>,
    tensor<10x512xf32>,
    tensor<10xf32>
  ) -> (
    tensor<512x784xf32>,
    tensor<512xf32>,
    tensor<512x512xf32>,
    tensor<512xf32>,
    tensor<10x512xf32>,
    tensor<10xf32>
  )
  return %res#0, %res#1, %res#2, %res#3, %res#4, %res#5 :
    tensor<512x784xf32>,
    tensor<512xf32>,
    tensor<512x512xf32>,
    tensor<512xf32>,
    tensor<10x512xf32>,
    tensor<10xf32>
}
