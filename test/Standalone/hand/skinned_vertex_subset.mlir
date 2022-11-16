// The subset of get_skinned_vertex_positions outside of get_posed_relatives, relatives_to_absolutes,
// and the batch matrix multiplication, but before applying global transform.
// This appears to be relevant because of correctness issues surrounding choosing to cache
// certain tensor iter_arguments.

func.func @skinned_vertex_subset(%transforms: tensor<22x4x4xf64>, %base_positions: tensor<544x4xf64>, %weights: tensor<544x22xf64>) -> tensor<544x3xf64> {
  %positions_init = arith.constant dense<0.0> : tensor<544x3xf64>
  %zero = arith.constant 0.0 : f64
  %curr_positions_init = arith.constant dense<0.0> : tensor<544x4xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cb = arith.constant 22 : index
  %positions = scf.for %iv = %c0 to %cb step %c1 iter_args(%positions_i = %positions_init) -> tensor<544x3xf64> {
    %transforms_slice = tensor.extract_slice %transforms[%iv, 0, 0] [1, 4, 4] [1, 1, 1] : tensor<22x4x4xf64> to tensor<4x4xf64>
    %curr_positions = linalg.generic
      {
        doc = "Column-major matrix multiplication",
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d2, d0)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d0)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      }
      ins(%transforms_slice, %base_positions : tensor<4x4xf64>, tensor<544x4xf64>)
      outs(%curr_positions_init : tensor<544x4xf64>) {
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      %0 = arith.mulf %arg0, %arg1 : f64
      %1 = arith.addf %0, %arg2 : f64
      linalg.yield %1 : f64
    } -> tensor<544x4xf64>
    %cp_slice = tensor.extract_slice %curr_positions[0, 0] [544, 3] [1, 1] : tensor<544x4xf64> to tensor<544x3xf64>
    %weight_slice = tensor.extract_slice %weights[0, %iv] [544, 1] [1, 1] : tensor<544x22xf64> to tensor<544xf64>

    %positions_next = linalg.generic
      {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%cp_slice, %weight_slice : tensor<544x3xf64>, tensor<544xf64>)
      outs(%positions_i : tensor<544x3xf64>) {
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
      %0 = arith.mulf %arg0, %arg1 : f64
      %1 = arith.addf %0, %arg2 : f64
      linalg.yield %1 : f64
    } -> tensor<544x3xf64>
    scf.yield %positions_next : tensor<544x3xf64>
  }
  return %positions : tensor<544x3xf64>
}

func.func @lagrad_skinned_vertex_subset(%transforms: tensor<22x4x4xf64>, %base_positions: tensor<544x4xf64>, %weights: tensor<544x22xf64>) -> tensor<22x4x4xf64> {
  %f = constant @skinned_vertex_subset : (tensor<22x4x4xf64>, tensor<544x4xf64>, tensor<544x22xf64>) -> tensor<544x3xf64>
  %df = standalone.grad %f {of = [0]} :
    (tensor<22x4x4xf64>, tensor<544x4xf64>, tensor<544x22xf64>) -> tensor<544x3xf64>,
    (tensor<22x4x4xf64>, tensor<544x4xf64>, tensor<544x22xf64>) -> tensor<22x4x4xf64>
  %res = call_indirect %df(%transforms, %base_positions, %weights) : (tensor<22x4x4xf64>, tensor<544x4xf64>, tensor<544x22xf64>) -> tensor<22x4x4xf64>
  return %res : tensor<22x4x4xf64>
}
