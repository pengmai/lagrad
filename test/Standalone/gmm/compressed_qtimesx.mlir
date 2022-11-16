func.func @Qtimesx(%Ltri_slice: tensor<10xf64>, %xcentered: tensor<5xf64>) -> tensor<5xf64> {
  %zero = arith.constant 0.0 : f64
  %trmv_space = tensor.empty() : tensor<5xf64>
  %trmv_init = linalg.fill ins(%zero: f64) outs(%trmv_space: tensor<5xf64>) -> tensor<5xf64>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cd = arith.constant 5 : index
  %out_1 = scf.for %iv = %c0 to %cd step %c1 iter_args(%out_iter_i = %trmv_init) -> tensor<5xf64> {
    %Lidx_0 = arith.muli %c2, %cd : index
    %Lidx_1 = arith.subi %Lidx_0, %iv : index
    %Lidx_2 = arith.subi %Lidx_1, %c1 : index
    %Lidx_3 = arith.muli %Lidx_2, %iv : index
    %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

    %iv_plus_1 = arith.addi %iv, %c1 : index
    %out_iter_i_next = scf.for %jv = %iv_plus_1 to %cd step %c1 iter_args(%out_iter = %out_iter_i) -> (tensor<5xf64>) {
      %Lidx_5 = arith.addi %Lidx_4, %jv : index
      %Lidx = arith.subi %Lidx_5, %iv_plus_1 : index
      %0 = tensor.extract %Ltri_slice[%Lidx] : tensor<10xf64>
      %1 = tensor.extract %xcentered[%iv] : tensor<5xf64>
      %2 = tensor.extract %out_iter[%jv] : tensor<5xf64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %3, %2 : f64
      %out_next = tensor.insert %4 into %out_iter[%jv] : tensor<5xf64>

      scf.yield %out_next : tensor<5xf64>
    }
    scf.yield %out_iter_i_next : tensor<5xf64>
  }
  return %out_1 : tensor<5xf64>
}

// func @print_i(%arg0: index) {
//   %u = memref.alloca() : memref<i64>
//   %0 = arith.index_cast %arg0 : index to i64
//   memref.store %0, %u[] : memref<i64>
//   %U = memref.cast %u : memref<i64> to memref<*xi64>
//   call @printMemrefI64(%U) : (memref<*xi64>) -> ()
//   return
// }

// Gradient of xcentered
func.func @gradQtimesx(%Ltri_slice: tensor<10xf64>, %xcentered: tensor<5xf64>) -> tensor<5xf64> {
  %dx_space = tensor.empty() : tensor<5xf64>
  %zero = arith.constant 0.0 : f64
  %dx_init = linalg.fill ins(%zero: f64) outs(%dx_space: tensor<5xf64>) -> tensor<5xf64>
  %g = arith.constant dense<1.0> : tensor<5xf64>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %cd = arith.constant 5 : index
  %res = scf.for %raw_iv = %c0 to %cd step %c1 iter_args(%dx_outer = %dx_init) -> tensor<5xf64> {
    %iv = arith.subi %c4, %raw_iv : index
    %Lidx_0 = arith.muli %c2, %cd : index
    %Lidx_1 = arith.subi %Lidx_0, %iv : index
    %Lidx_2 = arith.subi %Lidx_1, %c1 : index
    %Lidx_3 = arith.muli %Lidx_2, %iv : index
    %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

    %iv_plus_1 = arith.addi %iv, %c1 : index
    %dout_iter_i_next = scf.for %jv = %iv_plus_1 to %cd step %c1 iter_args(%dx = %dx_outer) -> (tensor<5xf64>) {
      %idx_0 = arith.subi %cd, %jv : index
      %idx_1 = arith.addi %idx_0, %iv_plus_1 : index
      %idx = arith.subi %idx_1, %c1 : index

      %dbg0 = arith.addi %Lidx_4, %idx : index
      %Lidx = arith.subi %dbg0, %iv_plus_1 : index
      // %Lidx_dbg = arith.addi %Lidx_4, %idx : index
      // %Lidx = memref.load %cache[%idx] : memref<5xindex>
      // %dbg0 = arith.subi %idx_0, %c1 : index
      // %dbg1 = arith.addi %dbg0, %Lidx_4 : index
      // call @print_i(%dbg1) : (index) -> ()
      %0 = tensor.extract %Ltri_slice[%Lidx] : tensor<10xf64>
      %1 = tensor.extract %g[%jv] : tensor<5xf64>
      %2 = tensor.extract %dx[%iv] : tensor<5xf64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %3, %2 : f64
      %out_next = tensor.insert %4 into %dx[%iv] : tensor<5xf64>
      scf.yield %out_next : tensor<5xf64>
    }
    scf.yield %dout_iter_i_next : tensor<5xf64>
  }
  return %res : tensor<5xf64>
}

// Gradient of Ltri_slice
// func @gradQtimesx(%Ltri_slice: tensor<10xf64>, %xcentered: tensor<5xf64>) -> tensor<10xf64> {
//   %dLtri_init = linalg.init_tensor [10] : tensor<10xf64>
//   %zero = arith.constant 0.0 : f64
//   // %dLtri_init = linalg.fill(%zero, %dLtri_space) : f64, tensor<10xf64> -> tensor<10xf64>

//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//   %c4 = arith.constant 4 : index
//   %cd = arith.constant 5 : index
//   %dLtri = scf.for %raw_iv = %c0 to %cd step %c1 iter_args(%dLtri = %dLtri_init) -> tensor<10xf64> {
//     %iv = arith.subi %c4, %raw_iv : index
//     %Lidx_0 = arith.muli %c2, %cd : index
//     %Lidx_1 = arith.subi %Lidx_0, %iv : index
//     %Lidx_2 = arith.subi %Lidx_1, %c1 : index
//     %Lidx_3 = arith.muli %Lidx_2, %iv : index
//     %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

//     %iv_plus_1 = arith.addi %iv, %c1 : index
//     %dout_iter_i_next = scf.for %jv = %iv_plus_1 to %cd step %c1 iter_args(%dLtri_it = %dLtri) -> (tensor<10xf64>) {
//       %idx_0 = arith.subi %cd, %jv : index
//       %idx_1 = arith.addi %idx_0, %iv_plus_1 : index
//       %idx = arith.subi %idx_1, %c1 : index

//       %Lidx_5 = arith.addi %Lidx_4, %idx : index
//       %Lidx = arith.subi %Lidx_5, %iv_plus_1 : index
//       // call @print_i(%Lidx) : (index) -> ()
//       %1 = tensor.extract %xcentered[%iv] : tensor<5xf64>
//       %out_next = tensor.insert %1 into %dLtri_it[%Lidx] : tensor<10xf64>
//       scf.yield %out_next : tensor<10xf64>
//     }
//     scf.yield %dout_iter_i_next : tensor<10xf64>
//   }
//   return %dLtri : tensor<10xf64>
// }

func.func private @printMemrefF64(tensor<*xf64>) attributes { llvm.emit_c_interface }
func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

func.func @main() {
  %Ltri_slice = arith.constant dense<[1.2, 4.3, -2.1, 0.0, -5.3, 1.1, 10.6, 4.3, -8.7, 9.1]> : tensor<10xf64>
  %xcentered = arith.constant dense<[5.3, 1.9, 4.4, -10.1, 4.3]> : tensor<5xf64>
  %f = constant @Qtimesx : (tensor<10xf64>, tensor<5xf64>) -> tensor<5xf64>
  // %df = standalone.grad %f {of = [0]} : (tensor<10xf64>, tensor<5xf64>) -> tensor<5xf64>, (tensor<10xf64>, tensor<5xf64>) -> tensor<10xf64>
  %df = standalone.grad %f {of = [1]} : (tensor<10xf64>, tensor<5xf64>) -> tensor<5xf64>, (tensor<10xf64>, tensor<5xf64>) -> tensor<5xf64>
  %res = call_indirect %df(%Ltri_slice, %xcentered) : (tensor<10xf64>, tensor<5xf64>) -> tensor<5xf64>
  // %res = call @gradQtimesx(%Ltri_slice, %xcentered) : (tensor<10xf64>, tensor<5xf64>) -> tensor<5xf64>
  %U = tensor.cast %res : tensor<5xf64> to tensor<*xf64>
  call @printMemrefF64(%U) : (tensor<*xf64>) -> ()
  return
}
