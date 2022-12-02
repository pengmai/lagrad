func @func_with_const_memref(%arg0: memref<2xf64>, %arg1: memref<f64>, %out: memref<f64>) -> f64 {
  linalg.generic
    {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    }
    ins(%arg0, %arg1 : memref<2xf64>, memref<f64>)
    outs(%out : memref<f64>) {
  ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
    %0 = arith.mulf %arg2, %arg3 : f64
    %1 = arith.addf %0, %arg4 : f64
    linalg.yield %1 : f64
  }
  %ret = arith.constant 0.0 : f64
  return %ret : f64
}

func private @print_memref_f64(memref<*xf64>) attributes {llvm.emit_c_interface}

func @main() -> i64 {
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %targ0 = arith.constant dense<[1.2, -5.4]> : tensor<2xf64>
  %targ1 = arith.constant dense<4.3> : tensor<f64>
  %arg0 = memref.buffer_cast %targ0 : memref<2xf64>
  %arg1 = memref.buffer_cast %targ1 : memref<f64>
  %darg0 = memref.alloca() : memref<2xf64>
  linalg.fill(%zero, %darg0) : f64, memref<2xf64>

  %out = memref.alloca() : memref<f64>
  memref.store %zero, %out[] : memref<f64>
  %dout = memref.alloca() : memref<f64>
  memref.store %one, %dout[] : memref<f64>

  %f = constant @func_with_const_memref : (memref<2xf64>, memref<f64>, memref<f64>) -> f64
  %df = lagrad.diff %f {const = [1]} : (memref<2xf64>, memref<f64>, memref<f64>) -> f64, (memref<2xf64>, memref<2xf64>, memref<f64>, memref<f64>, memref<f64>) -> f64

  call_indirect %df(%arg0, %darg0, %arg1, %out, %dout) : (memref<2xf64>, memref<2xf64>, memref<f64>, memref<f64>, memref<f64>) -> f64
  %U = memref.cast %darg0 : memref<2xf64> to memref<*xf64>
  call @print_memref_f64(%U) : (memref<*xf64>) -> ()
  %ret = arith.constant 0 : i64
  return %ret : i64
}
