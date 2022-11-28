// Though simple, this load/store pattern in MLIR is giving Enzyme some trouble.
// The goal here is to find the simplest example that will yield an activity analysis error.

// TRY THIS WITH O2, the Enzyme method
func @loadstore(%arg0: memref<f32>) -> f32 {
  %loc = memref.alloc() : memref<f32>
  %arg0_val = memref.load %arg0[] : memref<f32>
  memref.store %arg0_val, %loc[] : memref<f32>

  %loc_val = memref.load %loc[] : memref<f32>
  %cst = arith.constant 2.3 : f32
  %final = arith.mulf %loc_val, %cst : f32
  memref.dealloc %loc : memref<f32>
  return %final : f32
}

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @main() -> i64 {
  %source_A = arith.constant dense<3.2> : tensor<f32>
  %zero = arith.constant 0.0 : f32

  %A = memref.buffer_cast %source_A : memref<f32>
  %dA = memref.alloca() : memref<f32>
  linalg.fill(%zero, %dA) : f32, memref<f32>

  %f = constant @loadstore : (memref<f32>) -> f32
  %df = standalone.diff %f : (memref<f32>) -> f32, (memref<f32>, memref<f32>) -> f32
  call_indirect %df(%A, %dA) : (memref<f32>, memref<f32>) -> f32

  %U = memref.cast %dA : memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  %ret = arith.constant 0 : i64
  return %ret : i64
}
