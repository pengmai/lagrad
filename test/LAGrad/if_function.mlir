func @square(%arg0: f64) -> f64{
  %0 = arith.mulf %arg0, %arg0 : f64
  return %0 : f64
}

func @ifelsefunc(%arg0: f64) -> f64 {
  %cst = arith.constant 0.0 : f64
  %0 = arith.cmpf "oge", %arg0, %cst : f64
  %1 = scf.if %0 -> f64 {
    %2 = call @square(%arg0) : (f64) -> f64
    %3 = arith.addf %2, %cst : f64
    scf.yield %3 : f64
  } else {
    scf.yield %arg0 : f64
  }
  return %1 : f64
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
  %arg = arith.constant 4.2 : f64
  %res = lagrad.grad @ifelsefunc(%arg) : (f64) -> f64
  call @print(%res) : (f64) -> ()
  return
}