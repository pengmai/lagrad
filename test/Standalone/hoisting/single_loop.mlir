func @single_loop() -> memref<100xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cn = arith.constant 1000 : index
  %parent_buf = memref.alloc() : memref<100xf64>
  %0 = scf.for %iv = %c0 to %cn step %c1 iter_args(%m_iter = %parent_buf) -> memref<100xf64> {
    %buf = memref.alloc() : memref<100xf64>
    scf.yield %buf : memref<100xf64>
  }
  return %0 : memref<100xf64>
}
