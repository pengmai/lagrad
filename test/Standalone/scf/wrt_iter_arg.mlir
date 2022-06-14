// Inspired by LSTMs, this function computes an adjoint w.r.t. a value that is carried through
// the loop iter args.

func @wrt_iter_arg(%state_outer: tensor<2x3xf64>) -> tensor<3xf64> {
  %cst = arith.constant dense<[1., 2., 3.]> : tensor<3xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %res:2 = scf.for %iv = %c0 to %c2 step %c1 iter_args(%x = %cst, %state = %state_outer) -> (tensor<3xf64>, tensor<2x3xf64>) {
    %slice = tensor.extract_slice %state[%iv, 0] [1, 3] [1, 1] : tensor<2x3xf64> to tensor<3xf64>
    %mul = arith.mulf %slice, %x : tensor<3xf64>
    %new_state = tensor.insert_slice %mul into %state[%iv, 0] [1, 3] [1, 1] : tensor<3xf64> into tensor<2x3xf64>
    %add = arith.addf %mul, %x : tensor<3xf64>
    scf.yield %add, %new_state : tensor<3xf64>, tensor<2x3xf64>
  }
  return %res#0 : tensor<3xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %state_outer = arith.constant dense<[[2., 2., 2.], [3., 2., 2.]]> : tensor<2x3xf64>
  // %primal = call @wrt_iter_arg(%state_outer) : (tensor<2x3xf64>) -> tensor<3xf64>
  // %U = tensor.cast %primal : tensor<3xf64> to tensor<*xf64>
  %f = constant @wrt_iter_arg : (tensor<2x3xf64>) -> tensor<3xf64>
  %df = standalone.grad %f {of = [0]} : (tensor<2x3xf64>) -> tensor<3xf64>, (tensor<2x3xf64>) -> tensor<2x3xf64>
  %res = call_indirect %df(%state_outer) : (tensor<2x3xf64>) -> tensor<2x3xf64>
  %U = tensor.cast %res : tensor<2x3xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
#map0 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map1 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s0 + d1 * s2 + s1)>
#map2 = affine_map<(d0) -> (d0)>

// func @refgrad_wrt_iter_arg(%arg0: tensor<2x3xf64>) -> tensor<2x3xf64> {
//   %cst = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//   %0 = arith.subi %c2, %c0 : index
//   %1 = memref.alloc(%0) : memref<?x3xf64>
//   %2 = memref.alloc(%0) : memref<?x2x3xf64>
//   %3:2 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %cst, %arg3 = %arg0) -> (tensor<3xf64>, tensor<2x3xf64>) {
//     %9 = memref.subview %1[%arg1, 0] [1, 3] [1, 1] : memref<?x3xf64> to memref<3xf64, #map0>
//     %10 = memref.buffer_cast %arg2 : memref<3xf64>
//     linalg.copy(%10, %9) : memref<3xf64>, memref<3xf64, #map0>
//     %11 = memref.subview %2[%arg1, 0] [1, 2] [1, 1] : memref<?x2x3xf64> to memref<2x3xf64, #map1>
//     %12 = memref.buffer_cast %arg3 : memref<2x3xf64>
//     linalg.copy(%12, %11) : memref<2x3xf64>, memref<2x3xf64, #map1>
//     %13 = tensor.extract_slice %arg3[%arg1, 0] [1, 3] [1, 1] : tensor<2x3xf64> to tensor<3xf64>
//     %14 = arith.mulf %13, %arg2 : tensor<3xf64>
//     %15 = tensor.insert_slice %14 into %arg3[%arg1, 0] [1, 3] [1, 1] : tensor<3xf64> into tensor<2x3xf64>
//     %16 = arith.addf %14, %arg2 : tensor<3xf64>
//     scf.yield %16, %15 : tensor<3xf64>, tensor<2x3xf64>
//   }
//   %cst_0 = arith.constant 1.000000e+00 : f64
//   %4 = linalg.init_tensor [3] : tensor<3xf64>
//   %5 = linalg.fill(%cst_0, %4) : f64, tensor<3xf64> -> tensor<3xf64>
//   %cst_1 = arith.constant 0.000000e+00 : f64
//   %6 = linalg.init_tensor [2, 3] : tensor<2x3xf64>
//   %7 = linalg.fill(%cst_1, %6) : f64, tensor<2x3xf64> -> tensor<2x3xf64>
//   %8:3 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %5, %arg3 = %5, %arg4 = %7) -> (tensor<3xf64>, tensor<3xf64>, tensor<2x3xf64>) {
//     %9 = arith.subi %c2, %arg1 : index
//     %c1_2 = arith.constant 1 : index
//     %10 = arith.addi %9, %c0 : index
//     %11 = arith.subi %10, %c1_2 : index
//     %12 = memref.subview %2[%11, 0] [1, 2] [1, 1] : memref<?x2x3xf64> to memref<2x3xf64, #map1>
//     %13 = memref.cast %12 : memref<2x3xf64, #map1> to memref<2x3xf64>
//     %14 = memref.tensor_load %13 : memref<2x3xf64>
//     %15 = memref.subview %1[%11, 0] [1, 3] [1, 1] : memref<?x3xf64> to memref<3xf64, #map0>
//     %16 = memref.cast %15 : memref<3xf64, #map0> to memref<3xf64>
//     %17 = memref.tensor_load %16 : memref<3xf64>
//     %18 = tensor.extract_slice %14[%11, 0] [1, 3] [1, 1] : tensor<2x3xf64> to tensor<3xf64>
//     %19 = arith.mulf %18, %17 : tensor<3xf64>
//     %20 = tensor.insert_slice %19 into %14[%11, 0] [1, 3] [1, 1] : tensor<3xf64> into tensor<2x3xf64>
//     %21 = arith.addf %19, %17 : tensor<3xf64>
//     %22 = arith.mulf %arg2, %17 : tensor<3xf64>
//     %23 = arith.mulf %arg2, %18 : tensor<3xf64>
//     %24 = linalg.generic {doc = "Add in place", indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%23 : tensor<3xf64>) outs(%arg2 : tensor<3xf64>) {
//     ^bb0(%arg5: f64, %arg6: f64):  // no predecessors
//       %28 = arith.addf %arg5, %arg6 : f64
//       linalg.yield %28 : f64
//     } -> tensor<3xf64>
//     %cst_3 = arith.constant 0.000000e+00 : f64
//     %before = tensor.extract_slice %arg4[%11, 0] [1, 3] [1, 1] : tensor<2x3xf64> to tensor<3xf64>
//     %add = arith.addf %before, %22 : tensor<3xf64>
//     %27 = tensor.insert_slice %add into %arg4[%11, 0] [1, 3] [1, 1] : tensor<3xf64> into tensor<2x3xf64>
//     scf.yield %24, %arg3, %27 : tensor<3xf64>, tensor<3xf64>, tensor<2x3xf64>
//   }
//   return %8#2 : tensor<2x3xf64>
// }
// func private @print_memref_f64(tensor<*xf64>) attributes {llvm.emit_c_interface}
// func @main() {
//   %cst = arith.constant dense<[[2.000000e+00, 2.000000e+00, 2.000000e+00], [3.000000e+00, 2.000000e+00, 2.000000e+00]]> : tensor<2x3xf64>
//   %f = constant @wrt_iter_arg : (tensor<2x3xf64>) -> tensor<3xf64>
//   %f_0 = constant @refgrad_wrt_iter_arg : (tensor<2x3xf64>) -> tensor<2x3xf64>
//   %0 = call_indirect %f_0(%cst) : (tensor<2x3xf64>) -> tensor<2x3xf64>
//   %1 = tensor.cast %0 : tensor<2x3xf64> to tensor<*xf64>
//   call @print_memref_f64(%1) : (tensor<*xf64>) -> ()
//   return
// }