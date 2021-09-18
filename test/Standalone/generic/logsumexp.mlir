func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

func @logsumexp(%x: tensor<3x4xf32>) -> tensor<3xf32> {
  %out_init = constant dense<0.0> : tensor<3xf32>
  %prelog = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%x : tensor<3x4xf32>)
    outs(%out_init : tensor<3xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = math.exp %arg0 : f32
    %1 = addf %0, %arg1 : f32
    linalg.yield %1 : f32
  } -> tensor<3xf32>
  %out = math.log %prelog : tensor<3xf32>
  return %out : tensor<3xf32>
}

func @main() {
  %x = constant dense<{{data}}> : tensor<3x4xf32>
  %f = constant @logsumexp : (tensor<3x4xf32>) -> tensor<3xf32>
  %df = standalone.grad %f : (tensor<3x4xf32>) -> tensor<3xf32>, (tensor<3x4xf32>) -> tensor<3x4xf32>
  %res = call_indirect %df(%x) : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %U = tensor.cast %res : tensor<3x4xf32> to tensor<*xf32>
  call @print_memref_f32(%U) : (tensor<*xf32>) -> ()
  return
}
