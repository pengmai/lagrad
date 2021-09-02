//
// Based on the implementation from https://github.com/microsoft/ADBench/blob/994fbde50a3ee3c1edc7e7bcdb105470e63d7362/src/python/modules/PyTorch/gmm_objective.py
//

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// func @gmm_objective(
//   %alphas: tensor<{{k}}xf32>,
//   %means: tensor<{{k}}x{{d}}xf32>,
//   %Qs: tensor<{{k}}x{{d}}xf32>,
//   %Ls: tensor<{{k}}x{{(d * (d-1) / 2) | int}}xf32>,
//   // %icf: tensor<{{k}}x{{(d + d * (d-1) / 2) | int}}xf32>,
//   %x: tensor<{{n}}x{{d}}xf32>,
//   %wishart_gamma: f32,
//   %wishart_m: i64
// ) -> f32 {
//   %Qdiags = math.exp %Qs : tensor<{{k}}x{{d}}xf32>
//   %sum_q_space = constant dense<0.0> : tensor<{{k}}xf32>
//   // Sum along the columns of the Q matrix
//   %sum_qs = linalg.generic
//     {
//       indexing_maps = [#map0, #map1],
//       iterator_types = ["parallel", "reduction"]
//     }
//     ins(%Qdiags: tensor<{{k}}x{{d}}xf32>)
//     outs(%sum_q_space: tensor<{{k}}xf32>) {
//   ^bb0(%arg0: f32, %arg1: f32):
//     %0 = addf %arg0, %arg1 : f32
//     linalg.yield %0 : f32
//   } -> tensor<{{k}}xf32>

//   %Ls_space = constant dense<0.0> : tensor<{{k}}x{{d}}x{{d}}xf32>
//   // %Ls = linalg.generic
//   //   {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
//   //   ins(%)
//   %0 = constant 0.0 : f32
//   return %0 : f32
// }

func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }

#map_all = affine_map<(d0, d1) -> (d0, d1)>
#map_diag = affine_map<(d0, d1) -> (d0)>
func @main() {
  // How do I implement an np.diag using MLIR?
  %cst0 = constant 0.0 : f32
  %cst1 = constant 1.0 : f32
  %diag = constant dense<[0.1, 0.4]> : tensor<2xf32>
  %out_space = constant dense<0.0> : tensor<2x2xf32>
  // %U0 = tensor.cast %lower_tri : tensor<2x2xf32> to tensor<*xf32>
  // call @print_memref_f32(%U0) : (tensor<*xf32>) -> ()
  // %outM = memref.alloc() : memref<2x2xf32>
  // linalg.fill(%cst0, %outM) : f32, memref<2x2xf32>
  // %out_space tensor.load

  %out = tensor.generate {
  ^bb0(%i : index, %j : index):
    %0 = cmpi "eq", %i, %j : index
    %1 = scf.if %0 -> f32 {
      %2 = tensor.extract %diag[%i] : tensor<2xf32>
      scf.yield %2 : f32
    } else {
      scf.yield %cst0 : f32
    }
    tensor.yield %1 : f32
  } : tensor<2x2xf32>

  // %out = linalg.generic
  //   {indexing_maps = [#map_diag, #map_all], iterator_types = ["parallel", "parallel"]}
  //   ins(%diag : tensor<2xf32>)
  //   outs(%out_space : tensor<2x2xf32>) {
  // ^bb0(%arg0: f32, %arg1: f32):
  //   %0 = addf %arg0, %arg1 : f32
  //   linalg.yield %0 : f32
  // } -> tensor<2x2xf32>
  %U = tensor.cast %out : tensor<2x2xf32> to tensor<*xf32>
  call @print_memref_f32(%U) : (tensor<*xf32>) -> ()
  return
}
