#map = affine_map<(d0) -> (d0)>

func @cross_entropy(%activations: tensor<4xf64>, %label: index) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cd = arith.constant 4 : index
  %max_init = tensor.extract %activations[%c0] : tensor<4xf64>
  %max = scf.for %iv = %c1 to %cd step %c1 iter_args(%max_it = %max_init) -> f64 {
    %ai = tensor.extract %activations[%iv] : tensor<4xf64>
    %p = arith.cmpf ogt, %ai, %max_it : f64
    %max_next = select %p, %ai, %max_it : f64
    scf.yield %max_next : f64
  }
  %zerot = arith.constant dense<0.0> : tensor<4xf64>

  // Watch out for unnecessary copies here
  %exp = linalg.generic
    {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel"]
    }
    ins(%activations : tensor<4xf64>)
    outs(%zerot : tensor<4xf64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = arith.subf %arg0, %max : f64
    %1 = math.exp %0 : f64
    linalg.yield %1 : f64
  } -> tensor<4xf64>

  %zero = arith.constant 0.0 : f64
  %sum = scf.for %iv = %c0 to %cd step %c1 iter_args(%sum_it = %zero) -> (f64) {
    %exp_i = tensor.extract %exp[%iv] : tensor<4xf64>
    %sum_next = arith.addf %sum_it, %exp_i : f64
    scf.yield %sum_next : f64
  }

  %cross_entropy = linalg.generic
    {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel"]
    }
    ins(%exp : tensor<4xf64>)
    outs(%zerot : tensor<4xf64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = arith.divf %arg0, %sum : f64
    linalg.yield %0 : f64
  } -> tensor<4xf64>
  %probability = tensor.extract %cross_entropy[%label] : tensor<4xf64>
  %logprob = math.log %probability : f64
  %nlogprob = arith.negf %logprob : f64
  return %nlogprob : f64
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %x = arith.constant dense<[-4., 0.6, -1.2, 0.12]> : tensor<4xf64>
  %label = arith.constant 1 : index
  %f = constant @cross_entropy : (tensor<4xf64>, index) -> f64
  %df = standalone.grad %f : (tensor<4xf64>, index) -> f64, (tensor<4xf64>, index) -> tensor<4xf64>
  %res = call_indirect %df(%x, %label) : (tensor<4xf64>, index) -> tensor<4xf64>
  // %res = call @cross_entropy(%x, %label) : (tensor<4xf64>, index) -> f64
  // %s = linalg.init_tensor [] : tensor<f64>
  // %s1 = tensor.insert %res into %s[] : tensor<f64>
  // %U = tensor.cast %s1 : tensor<f64> to tensor<*xf64>
  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
