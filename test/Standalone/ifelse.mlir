func.func @ifelse(%arg0: f64) -> f64 {
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
func.func @ifelse_only_then_block(%arg0: f64) -> f64 {
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

func.func @ifsecondarg(%arg0: f64, %arg1: f64) -> f64 {
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

func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }

func.func @print(%arg0: f64) {
  %m = memref.alloca() : memref<f64>
  memref.store %arg0, %m[] : memref<f64>
  %U = memref.cast %m : memref<f64> to memref<*xf64>
  call @printMemrefF64(%U) : (memref<*xf64>) -> ()
  return
}

func.func @main() {
  %arg = arith.constant 4.5 : f64
  %arg1 = arith.constant -1.0 : f64

  %f = constant @ifelse : (f64) -> f64
  %df = standalone.grad %f : (f64) -> f64, (f64) -> f64
  %res = call_indirect %df(%arg) : (f64) -> f64
  call @print(%res) : (f64) -> ()

  %res1 = call_indirect %df(%arg1) : (f64) -> f64
  call @print(%res1) : (f64) -> ()

  %f0 = constant @ifelse_only_then_block : (f64) -> f64
  %df0 = standalone.grad %f0 : (f64) -> f64, (f64) -> f64
  %res2 = call_indirect %df0(%arg1) : (f64) -> f64
  call @print(%res2) : (f64) -> ()

  %f1 = constant @ifsecondarg : (f64, f64) -> f64
  %df1 = standalone.grad %f1 {of = [1]} : (f64, f64) -> f64, (f64, f64) -> f64
  %res3 = call_indirect %df1(%arg, %arg1) : (f64, f64) -> f64
  call @print(%res3) : (f64) -> ()

  %res4 = call_indirect %df1(%arg1, %arg) : (f64, f64) -> f64
  call @print(%res4) : (f64) -> ()
  return
}
