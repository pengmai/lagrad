func.func private @printMemrefF32(tensor<*xf32>) attributes { llvm.emit_c_interface }

func.func @logsumexp(%x: tensor<3x4xf32>) -> tensor<3xf32> {
  %out_init = arith.constant dense<0.0> : tensor<3xf32>
  %prelog = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%x : tensor<3x4xf32>)
    outs(%out_init : tensor<3xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = math.exp %arg0 : f32
    %1 = arith.addf %0, %arg1 : f32
    linalg.yield %1 : f32
  } -> tensor<3xf32>
  %out = math.log %prelog : tensor<3xf32>
  return %out : tensor<3xf32>
}

func.func @main() {
  %x = arith.constant dense<{{data}}> : tensor<3x4xf32>
  %res = lagrad.grad @logsumexp(%x) : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %U = tensor.cast %res : tensor<3x4xf32> to tensor<*xf32>
  call @printMemrefF32(%U) : (tensor<*xf32>) -> ()
  return
}
