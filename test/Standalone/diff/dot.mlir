// func @dot(%A : memref<4xf32>, %B : memref<4xf32>) -> memref<f32> {
//   %init = memref.alloc() : memref<f32>
//   %zero = constant 0.0 : f32
//   memref.store %zero, %init[] : memref<f32>
//   linalg.dot ins(%A, %B : memref<4xf32>, memref<4xf32>) outs(%init : memref<f32>)
//   return %init : memref<f32>
// }

builtin.func @dot(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %loc = memref.alloca() : memref<1xf32>
  %arg0_val = memref.load %arg0[%c0] : memref<4xf32>
  %arg1_val = memref.load %arg1[%c0] : memref<4xf32>
  %sum = addf %arg0_val, %arg1_val : f32

  memref.store %sum, %loc[%c0] : memref<1xf32>

  %1 = memref.load %arg0[%c1] : memref<4xf32>
  %2 = memref.store %2
  // %final = memref.load %loc[%c0] : memref<1xf32>
  return %sum : f32
}

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @main() {
  %source_A = constant dense<[1., 2., 3., 4.]> : tensor<4xf32>
  %source_B = constant dense<[9., 10., -11., 12.]> : tensor<4xf32>
  %zero = constant 0.0 : f32

  %A = memref.buffer_cast %source_A : memref<4xf32>
  %dA = memref.alloca() : memref<4xf32>
  linalg.fill(%zero, %dA) : f32, memref<4xf32>
  %B = memref.buffer_cast %source_B : memref<4xf32>
  %dB = memref.alloca() : memref<4xf32>
  linalg.fill(%zero, %dB) : f32, memref<4xf32>

  %f = constant @dot : (memref<4xf32>, memref<4xf32>) -> f32
  %df = standalone.diff %f : (memref<4xf32>, memref<4xf32>) -> f32, (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> f32
  call_indirect %df(%A, %dA, %B, %dB) : (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> f32

  %U = memref.cast %dA : memref<4xf32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}
