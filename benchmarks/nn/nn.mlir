#map = affine_map<(d0) -> (d0)>
#relu = {doc = "ReLU", indexing_maps = [#map], iterator_types = ["parallel"]}

func @cross_entropy(%activations: tensor<10xf32>, %label: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cd = arith.constant 10 : index
  %max_init = tensor.extract %activations[%c0] : tensor<10xf32>
  %max = scf.for %iv = %c1 to %cd step %c1 iter_args(%max_it = %max_init) -> f32 {
    %ai = tensor.extract %activations[%iv] : tensor<10xf32>
    %p = arith.cmpf ogt, %ai, %max_it : f32
    %max_next = select %p, %ai, %max_it : f32
    scf.yield %max_next : f32
  }
  %zerot = arith.constant dense<0.0> : tensor<10xf32>

  // Watch out for unnecessary copies here
  %exp = linalg.generic
    {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel"]
    }
    ins(%activations : tensor<10xf32>)
    outs(%zerot : tensor<10xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = arith.subf %arg0, %max : f32
    %1 = math.exp %0 : f32
    linalg.yield %1 : f32
  } -> tensor<10xf32>

  %zero = arith.constant 0.0 : f32
  %sum = scf.for %iv = %c0 to %cd step %c1 iter_args(%sum_it = %zero) -> (f32) {
    %exp_i = tensor.extract %exp[%iv] : tensor<10xf32>
    %sum_next = arith.addf %sum_it, %exp_i : f32
    scf.yield %sum_next : f32
  }

  %cross_entropy = linalg.generic
    {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel"]
    }
    ins(%exp : tensor<10xf32>)
    outs(%zerot : tensor<10xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = arith.divf %arg0, %sum : f32
    linalg.yield %0 : f32
  } -> tensor<10xf32>
  %probability = tensor.extract %cross_entropy[%label] : tensor<10xf32>
  %logprob = math.log %probability : f32
  %nlogprob = arith.negf %logprob : f32
  return %nlogprob : f32
}

func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }
func private @print_memref_i32(tensor<*xi32>) attributes { llvm.emit_c_interface }

func @mlir_mlp(
  %input: tensor<64x784xf32>,
  %labels: tensor<64xi32>,
  %w0: tensor<512x784xf32>,
  %b0: tensor<512xf32>,
  %w1: tensor<512x512xf32>,
  %b1: tensor<512xf32>,
  %w2: tensor<10x512xf32>,
  %b2: tensor<10xf32>
) -> f32 {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cb = arith.constant 64 : index
  %loss = scf.for %bv = %c0 to %cb step %c1 iter_args(%loss_it = %zero) -> f32 {
    %input_slice = tensor.extract_slice %input[%bv, 0] [1, 784] [1, 1] : tensor<64x784xf32> to tensor<784xf32>
    %h0_0 = linalg.matvec ins(%w0, %input_slice : tensor<512x784xf32>, tensor<784xf32>) outs(%b0 : tensor<512xf32>) -> tensor<512xf32>
    %h0 = linalg.generic #relu outs(%h0_0 : tensor<512xf32>) {
    ^bb0(%arg0: f32):
      %p = arith.cmpf ogt, %arg0, %zero : f32
      %0 = select %p, %arg0, %zero : f32
      linalg.yield %0 : f32
    } -> tensor<512xf32>

    %h1_0 = linalg.matvec ins(%w1, %h0 : tensor<512x512xf32>, tensor<512xf32>) outs(%b1 : tensor<512xf32>) -> tensor<512xf32>
    %h1 = linalg.generic #relu outs(%h1_0 : tensor<512xf32>) {
    ^bb0(%arg0: f32):
      %p = arith.cmpf ogt, %arg0, %zero : f32
      %0 = select %p, %arg0, %zero : f32
      linalg.yield %0 : f32
    } -> tensor<512xf32>

    %activations = linalg.matvec ins(%w2, %h1 : tensor<10x512xf32>, tensor<512xf32>) outs(%b2 : tensor<10xf32>) -> tensor<10xf32>
    %label = tensor.extract %labels[%bv] : tensor<64xi32>
    %l_idx = arith.index_cast %label : i32 to index
    %nll = call @cross_entropy(%activations, %l_idx) : (tensor<10xf32>, index) -> f32
    %loss_next = arith.addf %loss_it, %nll : f32
    scf.yield %loss_next : f32
  }
  %batch_size = arith.constant 64.0 : f32
  %avg_loss = arith.divf %loss, %batch_size : f32
  return %avg_loss : f32
}

func @lagrad_mlp(
  %input: tensor<64x784xf32>,
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
  %f = constant @mlir_mlp : (
    tensor<64x784xf32>,
    tensor<64xi32>,
    tensor<512x784xf32>,
    tensor<512xf32>,
    tensor<512x512xf32>,
    tensor<512xf32>,
    tensor<10x512xf32>,
    tensor<10xf32>
  ) -> f32
  %df = standalone.grad %f {of = [2, 3, 4, 5, 6, 7]} : (
    tensor<64x784xf32>,
    tensor<64xi32>,
    tensor<512x784xf32>,
    tensor<512xf32>,
    tensor<512x512xf32>,
    tensor<512xf32>,
    tensor<10x512xf32>,
    tensor<10xf32>
  ) -> f32, (
    tensor<64x784xf32>,
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
  %res:6 = call_indirect %df(%input, %labels, %w0, %b0, %w1, %b1, %w2, %b2) : (
    tensor<64x784xf32>,
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
