// Example of a Relu on tensors using a linalg.generic op.
// mlir-opt test/Standalone/relu.mlir -tensor-constant-bufferize -tensor-bufferize -func-bufferize -finalizing-bufferize -convert-linalg-to-loops -convert-scf-to-std -convert-memref-to-llvm -convert-std-to-llvm

#map0 = affine_map<(d0) -> (d0)>
func.func @relu(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %out = arith.constant dense<0.0> : tensor<4xf32>
  %cst = arith.constant 0.0 : f32
  %res = linalg.generic
    { indexing_maps = [#map0, #map0], iterator_types = ["parallel"] }
    ins(%arg0 : tensor<4xf32>)
    outs(%out : tensor<4xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %pred = arith.cmpf "olt", %arg1, %cst : f32
    %el = arith.select %pred, %cst, %arg1 : f32
    // %el = scf.if %pred -> (f32) {
    //   scf.yield %cst : f32
    // } else {
    //   scf.yield %arg1 : f32
    // }
    linalg.yield %el : f32
  } -> tensor<4xf32>
  return %res : tensor<4xf32>
}

func.func private @printMemrefF32(tensor<*xf32>) attributes { llvm.emit_c_interface }

func.func @main() {
  %0 = arith.constant dense<[1.0, -2.0, 3.0, -4.0]> : tensor<4xf32>
  %1 = lagrad.grad @relu(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = tensor.cast %1 : tensor<4xf32> to tensor<*xf32>
  call @printMemrefF32(%2) : (tensor<*xf32>) -> ()
  return
}
