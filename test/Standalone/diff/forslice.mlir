// This test case is based off the scf/forslice example, but in Enzyme.
#map0 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module  {
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<0.000000e+00>
  func private @print_memref_f64(memref<*xf64>) attributes {llvm.emit_c_interface}
  func @forslice(%arg0: memref<4x5xf64>, %arg1: memref<5x5xf64>) -> f64 {
    %cst = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %0 = memref.alloc() : memref<5xf64>
    %1 = memref.get_global @__constant_xf64 : memref<f64>
    %2 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %cst) -> (f64) {
      %3 = scf.for %arg4 = %c0 to %c5 step %c1 iter_args(%arg5 = %0) -> (memref<5xf64>) {
        %7 = memref.subview %arg0[%arg2, 0] [1, 5] [1, 1] : memref<4x5xf64> to memref<5xf64, #map0>
        %8 = memref.cast %7 : memref<5xf64, #map0> to memref<5xf64>
        %9 = memref.load %arg1[%arg4] : memref<5xf64>
        %10 = memref.alloc() : memref<f64>
        memref.copy(%1, %10) : memref<f64>, memref<f64>
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%8 : memref<5xf64>) outs(%10 : memref<f64>) {
        ^bb0(%arg6: f64, %arg7: f64):  // no predecessors
          %12 = arith.mulf %arg6, %9 : f64
          %13 = arith.addf %12, %arg7 : f64
          linalg.yield %13 : f64
        }
        %11 = memref.load %10[] : memref<f64>
        memref.store %11, %arg5[%arg4] : memref<5xf64>
        scf.yield %arg5 : memref<5xf64>
      }
      %4 = memref.alloc() : memref<f64>
      memref.copy(%1, %4) : memref<f64>, memref<f64>
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%3 : memref<5xf64>) outs(%4 : memref<f64>) {
      ^bb0(%arg4: f64, %arg5: f64):  // no predecessors
        %7 = arith.addf %arg4, %arg5 : f64
        linalg.yield %7 : f64
      }
      %5 = memref.load %4[] : memref<f64>
      %6 = arith.addf %5, %arg3 : f64
      scf.yield %6 : f64
    }
    return %2 : f64
  }

  func @main() -> i64 {
    %At = arith.constant dense<[
      [ 0.,  1.,  2.,  3.,  4.],
      [ 5.,  6.,  7.,  8.,  9.],
      [10., 11., 12., 13., 14.],
      [15., 16., 17., 18., 19.]
    ]> : tensor<4x5xf64>
    %Bt = arith.constant dense<[0., 1., 2., 3., 4.]> : tensor<5xf64>
    %A = memref.buffer_cast %At : memref<4x5xf64>
    %B = memref.buffer_cast %Bt : memref<5xf64>
    %dA = memref.alloc() : memref<4x5xf64>
    %dB = memref.alloc() : memref<5xf64>
    %zero = arith.constant 0.0 : f64
    linalg.fill(%zero, %dA) : f64, memref<4x5xf64>
    linalg.fill(%zero, %dB) : f64, memref<5xf64>

    %f = constant @forslice : (memref<4x5xf64>, memref<5xf64>) -> f64
    %df = standalone.diff %f : (memref<4x5xf64>, memref<5xf64>) -> f64, (memref<4x5xf64>, memref<4x5xf64>, memref<5xf64>, memref<5xf64>) -> f64
    call_indirect %df(%A, %dA, %B, %dB) : (memref<4x5xf64>, memref<4x5xf64>, memref<5xf64>, memref<5xf64>) -> f64
    %U = memref.cast %dA : memref<4x5xf64> to memref<*xf64>
    call @print_memref_f64(%U) : (memref<*xf64>) -> ()
    %UB = memref.cast %dB : memref<5xf64> to memref<*xf64>
    call @print_memref_f64(%UB) : (memref<*xf64>) -> ()

    memref.dealloc %dA : memref<4x5xf64>
    memref.dealloc %dB : memref<5xf64>
    %ret = arith.constant 0 : i64
    return %ret : i64
  }
}
