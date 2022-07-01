func @ifdifferenttype(%arg0: f64) -> tensor<3xf64> {
  %cst = arith.constant dense<0.0> : tensor<3xf64>
  %cst_0 = arith.constant 0.0 : f64
  %0 = arith.cmpf "oge", %arg0, %cst_0 : f64

  %2 = linalg.generic
    {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%cst : tensor<3xf64>)
    outs(%cst : tensor<3xf64>) {
  ^bb0(%arg1: f64, %arg2: f64):
    %3 = math.exp %arg0 : f64
    linalg.yield %3 : f64
  } -> tensor<3xf64>
  %1 = select %0, %2, %cst : tensor<3xf64>
  return %1 : tensor<3xf64>
}

func private @print_memref_f64(memref<*xf64>) attributes { llvm.emit_c_interface }

func @print(%arg0: f64) {
  %m = memref.alloca() : memref<f64>
  memref.store %arg0, %m[] : memref<f64>
  %U = memref.cast %m : memref<f64> to memref<*xf64>
  call @print_memref_f64(%U) : (memref<*xf64>) -> ()
  return
}

func @main() {
  %arg = arith.constant 1.2 : f64
  %f = constant @ifdifferenttype : (f64) -> tensor<3xf64>
  %df = standalone.grad %f : (f64) -> tensor<3xf64>, (f64) -> f64
  %res = call_indirect %df(%arg) : (f64) -> f64
  call @print(%res) : (f64) -> ()
  return
}
