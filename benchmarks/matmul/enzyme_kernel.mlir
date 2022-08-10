func @emmatmul(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %out: memref<?x?xf64>) -> f64 {
  linalg.matmul ins(%A, %B : memref<?x?xf64>, memref<?x?xf64>) outs(%out : memref<?x?xf64>)
  %zero = arith.constant 0.0 : f64
  return %zero : f64
}

func @enzyme_mlir_matmul(%A: memref<?x?xf64>, %B: memref<?x?xf64>) -> memref<?x?xf64> {
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 1.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %Adim0 = memref.dim %A, %c0 : memref<?x?xf64>
  %Adim1 = memref.dim %A, %c1 : memref<?x?xf64>
  %Bdim1 = memref.dim %B, %c1 : memref<?x?xf64>
  %dA = memref.alloc(%Adim0, %Adim1) : memref<?x?xf64>
  %out = memref.alloc(%Adim0, %Bdim1) : memref<?x?xf64>
  %dout = memref.alloc(%Adim0, %Bdim1) : memref<?x?xf64>
  linalg.fill(%zero, %dA) : f64, memref<?x?xf64>
  linalg.fill(%zero, %out) : f64, memref<?x?xf64>
  linalg.fill(%one, %dout) : f64, memref<?x?xf64>

  %f = constant @emmatmul : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> f64
  %df = standalone.diff %f {const = [1]} :
    (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> f64,
    (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> f64
  call_indirect %df(%A, %dA, %B, %out, %dout) : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> f64
  memref.dealloc %out : memref<?x?xf64>
  memref.dealloc %dout : memref<?x?xf64>
  return %dA : memref<?x?xf64>
}
