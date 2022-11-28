func @broadcast_square(%a: tensor<f32>) -> tensor<4xf32> {
  %out_init = arith.constant dense<0.0> : tensor<4xf32>
  %out = linalg.generic
    {
      indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%a : tensor<f32>)
    outs(%out_init : tensor<4xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    linalg.yield %arg0 : f32
  } -> tensor<4xf32>
  return %out : tensor<4xf32>
}

// primal (bb):  (a, b) -> (ab)^2
// (a, b) -> 2 * ab * b
// adjoint (bb): (a, b) -> 1 * b * 2 * b -> 2b^2
func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

func @main() {
  %a = arith.constant dense<2.2> : tensor<f32>

  %res = lagrad.grad @broadcast_square(%a) : (tensor<f32>) -> tensor<f32>
  %U = tensor.cast %res : tensor<f32> to tensor<*xf32>
  call @print_memref_f32(%U) : (tensor<*xf32>) -> ()
  return
}
