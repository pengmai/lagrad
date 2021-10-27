func @mlir_trimatvec_dense_primal(%M: tensor<{{n}}x{{n}}xf64>, %x: tensor<{{n}}xf64>, %out: tensor<{{n}}xf64> {linalg.inplaceable = true}) -> tensor<{{n}}xf64> {
  %2 = linalg.matvec ins(%M, %x : tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>) outs(%out : tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  return %2 : tensor<{{n}}xf64>
}

func @mlir_trimatvec_dense_adjoint(%M: tensor<{{n}}x{{n}}xf64>, %x: tensor<{{n}}xf64>, %out: tensor<{{n}}xf64> {linalg.inplaceable = true}) -> (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>) {
  %f = constant @mlir_trimatvec_dense_primal : (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  %df = standalone.grad %f {of = [0, 1]} : (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>, (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>)
  %res:2 = call_indirect %df(%M, %x, %out) : (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>)
  return %res#0, %res#1 : tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>
}

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
func @mlir_trimatvec_tri_primal(%M: tensor<{{n}}x{{n}}xf64, "ltri">, %x: tensor<{{n}}xf64>, %out: tensor<{{n}}xf64> {linalg.inplaceable = true}) -> tensor<{{n}}xf64> {
  %res = linalg.generic
    {
      indexing_maps = [#map0, #map1, #map2],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%M, %x : tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>) outs(%out : tensor<{{n}}xf64>) {
  ^bb0(%arg3: f64, %arg4: f64, %arg5: f64):  // no predecessors
    %1 = arith.mulf %arg3, %arg4 : f64
    %2 = arith.addf %arg5, %1 : f64
    linalg.yield %2 : f64
  } -> tensor<{{n}}xf64>
  return %res : tensor<{{n}}xf64>
}

func @mlir_trimatvec_tri_adjoint(%M: tensor<{{n}}x{{n}}xf64, "ltri">, %x: tensor<{{n}}xf64>, %out: tensor<{{n}}xf64> {linalg.inplaceable = true}) -> (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>) {
  %f = constant @mlir_trimatvec_tri_primal : (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  %df = standalone.grad %f {of = [0, 1]} : (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>, (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>)
  %res:2 = call_indirect %df(%M, %x, %out) : (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>, tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>)
  return %res#0, %res#1 : tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>
}

func @mlir_trimatvec_compressed_primal(%icf: tensor<{{(n * (n - 1) / 2)|round|int}}xf64>, %x: tensor<{{n}}xf64>, %out: tensor<{{n}}xf64> {linalg.inplaceable = true}) -> tensor<{{n}}xf64> {
  %space = memref.alloc() : memref<{{n}}xf64>
  %outm = memref.buffer_cast %out : memref<{{n}}xf64>
  linalg.copy(%outm, %space) : memref<{{n}}xf64>, memref<{{n}}xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cn = arith.constant {{n}} : index
  scf.for %iv = %c0 to %cn step %c1 iter_args(%it = %c0) -> index {
    %it_next = scf.for %jv = %c0 to %iv step %c1 iter_args(%jt = %it) -> index {
      %0 = tensor.extract %icf[%jt] : tensor<{{(n * (n - 1) / 2)|round|int}}xf64>
      %1 = tensor.extract %x[%jv] : tensor<{{n}}xf64>
      %2 = memref.load %space[%iv] : memref<{{n}}xf64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %3, %2 : f64
      memref.store %4, %space[%iv] : memref<{{n}}xf64>

      %kt = arith.addi %jt, %c1 : index
      scf.yield %kt : index
    }
    scf.yield %it_next : index
  }
  %res = memref.tensor_load %space : memref<{{n}}xf64>
  return %res : tensor<{{n}}xf64>
}

func @mlir_trimatvec_compressed_adjoint(%icf: tensor<{{(n * (n - 1) / 2)|round|int}}xf64>, %x: tensor<{{n}}xf64>, %out: tensor<{{n}}xf64> {linalg.inplaceable = true}) -> (tensor<{{(n * (n - 1) / 2)|round|int}}xf64>, tensor<{{n}}xf64>) {
  %ones = arith.constant dense<1.0> : tensor<{{n}}xf64>
  %arg1_space = memref.alloc() : memref<{{(n * (n - 1) / 2)|round|int}}xf64>
  %zero = arith.constant 0.0 : f64
  linalg.fill(%zero, %arg1_space) : f64, memref<{{(n * (n - 1) / 2)|round|int}}xf64>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cn = arith.constant {{n}} : index
  scf.for %iv = %c0 to %cn step %c1 iter_args(%it = %c0) -> index {
    %it_next = scf.for %jv = %c0 to %iv step %c1 iter_args(%jt = %it) -> index {
      %0 = tensor.extract %ones[%iv] : tensor<{{n}}xf64>
      %1 = tensor.extract %x[%jv] : tensor<{{n}}xf64>
      %2 = memref.load %arg1_space[%jt] : memref<{{(n * (n - 1) / 2)|round|int}}xf64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %3, %2 : f64
      memref.store %4, %arg1_space[%jt] : memref<{{(n * (n - 1) / 2)|round|int}}xf64>

      %kt = arith.addi %jt, %c1 : index
      scf.yield %kt : index
    }
    scf.yield %it_next : index
  }

  %arg2_space = memref.alloc() : memref<{{n}}xf64>
  linalg.fill(%zero, %arg2_space) : f64, memref<{{n}}xf64>
  scf.for %iv = %c0 to %cn step %c1 iter_args(%it = %c0) -> index {
    %it_next = scf.for %jv = %c0 to %iv step %c1 iter_args(%jt = %it) -> index {
      %0 = tensor.extract %icf[%jt] : tensor<{{(n * (n - 1) / 2)|round|int}}xf64>
      %1 = tensor.extract %ones[%iv] : tensor<{{n}}xf64>
      %2 = memref.load %arg2_space[%jv] : memref<{{n}}xf64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %3, %2 : f64
      memref.store %4, %arg2_space[%jv] : memref<{{n}}xf64>

      %kt = arith.addi %jt, %c1 : index
      scf.yield %kt : index
    }
    scf.yield %it_next: index
  }

  %arg1_res = memref.tensor_load %arg1_space : memref<{{(n * (n - 1) / 2)|round|int}}xf64>
  %arg2_res = memref.tensor_load %arg2_space : memref<{{n}}xf64>
  return %arg1_res, %arg2_res : tensor<{{(n * (n - 1) / 2)|round|int}}xf64>, tensor<{{n}}xf64>
}
