#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
func.func @batched_cross_entropy(%activations: tensor<4x2xf32>, %labels: tensor<2xindex>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cd = arith.constant 4 : index
  %cb = arith.constant 2 : index
  %cbf = arith.constant 2.0 : f32
  %max_init = tensor.extract_slice %activations[0, 0] [1, 2] [1, 1] : tensor<4x2xf32> to tensor<2xf32>
  %max = scf.for %iv = %c1 to %cd step %c1 iter_args(%max_outer = %max_init) -> tensor<2xf32> {
    %max_outer_next = scf.for %bv = %c0 to %cb step %c1 iter_args(%max_it = %max_outer) -> tensor<2xf32> {
      %ai = tensor.extract %activations[%iv, %bv] : tensor<4x2xf32>
      %max_val = tensor.extract %max_it[%bv] : tensor<2xf32>
      %p = arith.cmpf ogt, %ai, %max_val : f32
      %max_val_next = arith.select %p, %ai, %max_val : f32
      %max_next = tensor.insert %max_val_next into %max_it[%bv] : tensor<2xf32>
      scf.yield %max_next : tensor<2xf32>
    }
    scf.yield %max_outer_next : tensor<2xf32>
  }

  %zerot = arith.constant dense<0.0> : tensor<4x2xf32>
  %zero2 = arith.constant dense<0.0> : tensor<2xf32>
  %exp = linalg.generic
    { indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"] }
    ins(%activations, %max : tensor<4x2xf32>, tensor<2xf32>)
    outs(%zerot : tensor<4x2xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.subf %arg0, %arg1 : f32
    %1 = math.exp %0 : f32
    linalg.yield %1 : f32
  } -> tensor<4x2xf32>

  %total = linalg.generic
    { indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel"] }
    ins(%exp : tensor<4x2xf32>)
    outs(%zero2 : tensor<2xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<2xf32>

  %zero = arith.constant 0.0 : f32
  %cross_entropy_total = scf.for %iv = %c0 to %cb step %c1 iter_args(%ce_it = %zero) -> f32 {
    %label = tensor.extract %labels[%iv] : tensor<2xindex>
    %exp_val = tensor.extract %exp[%label, %iv] : tensor<4x2xf32>
    %total_val = tensor.extract %total[%iv] : tensor<2xf32>
    %sm_val = arith.divf %exp_val, %total_val : f32
    %ll = math.log %sm_val : f32
    %nll = arith.negf %ll : f32
    %ce_next = arith.addf %ce_it, %nll : f32
    scf.yield %ce_next : f32
  }
  %primal = arith.divf %cross_entropy_total, %cbf : f32
  return %primal : f32
}

func.func private @print_f32(%x: f32) {
  %0 = tensor.empty() : tensor<f32>
  %1 = tensor.insert %x into %0[] : tensor<f32>
  %2 = tensor.cast %1 : tensor<f32> to tensor<*xf32>
  call @printMemrefF32(%2) : (tensor<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(tensor<*xf32>) attributes { llvm.emit_c_interface }
func.func @main() {
  %activations = arith.constant dense<[
    [-4.0,  5.0],
    [ 0.6,  4.0],
    [-1.2, -2.0],
    [0.12,  1.0]
  ]> : tensor<4x2xf32>
  %labels = arith.constant dense<[1, 2]> : tensor<2xindex>
  // %res = call @batched_cross_entropy(%activations, %labels) : (tensor<4x2xf32>, tensor<2xindex>) -> f32
  %f = constant @batched_cross_entropy : (tensor<4x2xf32>, tensor<2xindex>) -> f32
  %df = standalone.grad %f : (tensor<4x2xf32>, tensor<2xindex>) -> f32, (tensor<4x2xf32>, tensor<2xindex>) -> tensor<4x2xf32>
  %res = call_indirect %df(%activations, %labels) : (tensor<4x2xf32>, tensor<2xindex>) -> tensor<4x2xf32>
  %U = tensor.cast %res : tensor<4x2xf32> to tensor<*xf32>
  call @printMemrefF32(%U) : (tensor<*xf32>) -> ()
  return
}
