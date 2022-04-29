func @pow(%x: tensor<4xf64>, %i: index) -> tensor<4xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f64
  %p_space = linalg.init_tensor [4] : tensor<4xf64>
  %p_init = linalg.fill(%one, %p_space) : f64, tensor<4xf64> -> tensor<4xf64>
  %res = scf.for %iv = %c0 to %i step %c1 iter_args(%p = %p_init) -> tensor<4xf64> {
    %1 = arith.mulf %p, %x : tensor<4xf64>
    scf.yield %1 : tensor<4xf64>
  }
  return %res : tensor<4xf64>
}

// #map = affine_map<(d0) -> (d0)>
// #map0 = affine_map<(d0)[s0] -> (d0 + s0)>
// func @__grad_pow(%arg0: tensor<4xf64>, %arg1: index) -> tensor<4xf64> {
//   %cst = arith.constant 0.000000e+00 : f64
//   %cst_0 = arith.constant 1.000000e+00 : f64
//   %c1 = arith.constant 1 : index
//   %c0 = arith.constant 0 : index
//   %0 = linalg.init_tensor [4] : tensor<4xf64>
//   %1 = linalg.fill(%cst_0, %0) : f64, tensor<4xf64> -> tensor<4xf64>
//   %cache = memref.alloc(%arg1) : memref<?x4xf64> // what is the shape of the tensor?
//   %2 = scf.for %arg2 = %c0 to %arg1 step %c1 iter_args(%arg3 = %1) -> (tensor<4xf64>) {
//     // save in cache
//     %marg3 = memref.buffer_cast %arg3 : memref<4xf64>
//     %view = memref.subview %cache[%arg2, 0] [1, 4] [1, 1] : memref<?x4xf64> to memref<4xf64, #map0>
//     linalg.copy(%marg3, %view) : memref<4xf64>, memref<4xf64, #map0>

//     %8 = arith.mulf %arg3, %arg0 : tensor<4xf64>
//     scf.yield %8 : tensor<4xf64>
//   }
//   %3 = linalg.init_tensor [4] : tensor<4xf64>
//   %4 = linalg.fill(%cst_0, %3) : f64, tensor<4xf64> -> tensor<4xf64>
//   %5 = linalg.init_tensor [4] : tensor<4xf64>
//   %6 = linalg.fill(%cst, %5) : f64, tensor<4xf64> -> tensor<4xf64>
//   %7:2 = scf.for %arg2 = %c0 to %arg1 step %c1 iter_args(%arg3 = %4, %arg4 = %6) -> (tensor<4xf64>, tensor<4xf64>) {
//     %idx_0 = arith.subi %arg1, %arg2 : index
//     %idx = arith.subi %idx_0, %c1 : index
//     // read from cache
//     %view = memref.subview %cache[%idx, 0] [1, 4] [1, 1] : memref<?x4xf64> to memref<4xf64, #map0>
//     %temp = memref.cast %view : memref<4xf64, #map0> to memref<4xf64>
//     %primal_cached = memref.tensor_load %temp : memref<4xf64>
  
//     %8 = arith.mulf %arg3, %arg0 : tensor<4xf64>
//     %9 = arith.mulf %arg3, %primal_cached : tensor<4xf64>
//     %10 = linalg.generic {doc = "Add in place", indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%9 : tensor<4xf64>) outs(%arg4 : tensor<4xf64>) {
//     ^bb0(%arg5: f64, %arg6: f64):  // no predecessors
//       %11 = arith.addf %arg5, %arg6 : f64
//       linalg.yield %11 : f64
//     } -> tensor<4xf64>
//     scf.yield %8, %10 : tensor<4xf64>, tensor<4xf64>
//   }
//   return %7#1 : tensor<4xf64>
// }

func private @print_memref_f64(tensor<*xf64>) attributes {llvm.emit_c_interface}

func @main() {
  %arg = arith.constant dense<[1.3, 1.4, 1.5, 1.6]> : tensor<4xf64>
  %c4 = arith.constant 4 : index
  %f = constant @pow : (tensor<4xf64>, index) -> tensor<4xf64>
  %df = standalone.grad %f {of = [0]} : (tensor<4xf64>, index) -> tensor<4xf64>, (tensor<4xf64>, index) -> tensor<4xf64>

  %res = call_indirect %df(%arg, %c4) : (tensor<4xf64>, index) -> tensor<4xf64>
  // %res = call @__grad_pow(%arg, %c4) : (tensor<4xf64>, index) -> tensor<4xf64>

  %U = tensor.cast %res : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()

  // %x = arith.constant 1 : index
  // %c5 = arith.constant 5 : index
  // %cache = memref.alloca(%c5) : memref<?x4xf64>
  // %view = memref.subview %cache[%x, 0] [1, 4] [1, 1] : memref<?x4xf64> to memref<4xf64>
  return
}
