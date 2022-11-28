func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }
// func private @print_memref_f64(memref<*xf64>) attributes { llvm.emit_c_interface }
#map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
func @main() {
  %t = arith.constant dense<[
    [1., -1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [2., -2., 8., 2., 2., 2., 2., 2., 2., 2.],
    [3., -3., 3., 3., 3., 3., 3., 3., 3., 3.],
    [4., -4., 4., 4., 4., 4., 4., 4., 4., 4.]
  ]> : tensor<4x10xf64>

  %slice = tensor.extract_slice %t[0, 1] [4, 1] [1, 1] : tensor<4x10xf64> to tensor<4xf64>
  %U = tensor.cast %slice : tensor<4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()

  // %m = memref.buffer_cast %t : memref<4x10xf64>
  // %s = memref.subview %m [0, 2] [4, 1] [1, 1] : memref<4x10xf64> to memref<4xf64, #map>
  // %U = memref.cast %s : memref<4xf64, #map> to memref<*xf64>
  // %s = memref.subview %m [1, 0] [1, 10] [1, 1] : memref<4x10xf64> to memref<10xf64, #map>
  // %U = memref.cast %s : memref<10xf64, #map> to memref<*xf64>
  // call @print_memref_f64(%U) : (memref<*xf64>) -> ()
  return
}
