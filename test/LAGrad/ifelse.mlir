func @ifelse(%arg0: f64) -> f64 {
  %cst = arith.constant 0.0 : f64
  %0 = arith.cmpf "oge", %arg0, %cst : f64
  %1 = scf.if %0 -> f64 {
    %2 = arith.mulf %arg0, %arg0 : f64
    %3 = arith.addf %2, %cst : f64
    scf.yield %3 : f64
  } else {
    scf.yield %arg0 : f64
  }
  return %1 : f64
}

// The unbound variable only occurs in the then block, not the else block.
func @ifelse_only_then_block(%arg0: f64) -> f64 {
  %cst = arith.constant 0.0 : f64
  %0 = arith.cmpf "oge", %arg0, %cst : f64
  %1 = scf.if %0 -> f64 {
    %2 = arith.mulf %arg0, %arg0 : f64
    %3 = arith.addf %2, %cst : f64
    scf.yield %3 : f64
  } else {
    scf.yield %cst : f64
  }
  return %1 : f64
}

func @ifsecondarg(%arg0: f64, %arg1: f64) -> f64 {
  %cst = arith.constant 3.4 : f64
  %0 = arith.cmpf "oge", %arg0, %arg1 : f64
  %1 = arith.mulf %cst, %arg1 : f64
  %2 = scf.if %0 -> f64 {
    %3 = arith.mulf %arg0, %arg1: f64
    scf.yield %3 : f64
  } else {
    %3 = arith.mulf %1, %arg1 : f64
    %4 = arith.mulf %cst, %3 : f64
    scf.yield %4 : f64
  }
  %5 = arith.mulf %2, %cst : f64
  return %5 : f64
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
  %arg = arith.constant 4.5 : f64
  %arg1 = arith.constant -1.0 : f64

  %res = lagrad.grad @ifelse(%arg) : (f64) -> f64
  call @print(%res) : (f64) -> ()
  %res1 = lagrad.grad @ifelse(%arg1) : (f64) -> f64
  call @print(%res1) : (f64) -> ()
  %res2 = lagrad.grad @ifelse_only_then_block(%arg1) : (f64) -> f64
  call @print(%res2) : (f64) -> ()
  %res3 = lagrad.grad @ifsecondarg(%arg, %arg1) {of = [1]} : (f64, f64) -> f64
  call @print(%res3) : (f64) -> ()
  %res4 = lagrad.grad @ifsecondarg(%arg1, %arg) {of = [1]}: (f64, f64) -> f64
  call @print(%res4) : (f64) -> ()
  return
}
