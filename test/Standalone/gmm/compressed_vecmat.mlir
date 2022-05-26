// This test doesn't test LAGrad, only my handwritten implementation.
func @vecmat(%x: tensor<5xf64>, %L: tensor<10xf64>, %vm_init: tensor<5xf64>) -> tensor<5xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cd = arith.constant 5 : index
  %out = scf.for %iv = %c0 to %cd step %c1 iter_args(%outer = %vm_init) -> tensor<5xf64> {
    %Lidx_0 = arith.muli %c2, %cd : index
    %Lidx_1 = arith.subi %Lidx_0, %iv : index
    %Lidx_2 = arith.subi %Lidx_1, %c1 : index
    %Lidx_3 = arith.muli %Lidx_2, %iv : index
    %Lidx_4 = arith.divsi %Lidx_3, %c2 : index

    %iv_plus_1 = arith.addi %iv, %c1 : index
    %outer_next = scf.for %jv = %iv_plus_1 to %cd step %c1 iter_args(%out_iter = %outer) -> (tensor<5xf64>) {
      %Lidx_5 = arith.addi %Lidx_4, %jv : index
      %Lidx = arith.subi %Lidx_5, %iv_plus_1 : index
      %0 = tensor.extract %L[%Lidx] : tensor<10xf64>
      %1 = tensor.extract %x[%jv] : tensor<5xf64>
      %2 = tensor.extract %out_iter[%iv] : tensor<5xf64>
      %3 = arith.mulf %0, %1 : f64
      %4 = arith.addf %3, %2 : f64
      %out_next = tensor.insert %4 into %out_iter[%iv] : tensor<5xf64>
      scf.yield %out_next : tensor<5xf64>
    }
    scf.yield %outer_next : tensor<5xf64>
  }
  // %out = linalg.generic
  //   {
  //     indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>],
  //     iterator_types = ["parallel", "reduction"]
  //   }
  //   ins(%x : tensor<5xf64>)
  //   outs(%vm_init : tensor<5xf64>) {
  // ^bb0(%arg0: f64, %arg1: f64):
  //   %idx0 = linalg.index
  // } -> tensor<5xf64>
  return %out : tensor<5xf64>
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %zero = arith.constant 0.0 : f64
  %x = arith.constant dense<[1.2, -3.1, 6.5, -4.2, 1.2]> : tensor<5xf64>
  %L = arith.constant dense<[3.3, -3.2, 2.1, 1.2, 0.6, -3.2, 10.1, 1.1, 3.6, 4.5]> : tensor<10xf64>
  %vm_space = linalg.init_tensor [5] : tensor<5xf64>
  %vm_init = linalg.fill(%zero, %vm_space) : f64, tensor<5xf64> -> tensor<5xf64>
  %res = call @vecmat(%x, %L, %vm_init) : (tensor<5xf64>, tensor<10xf64>, tensor<5xf64>) -> tensor<5xf64>
  %U = tensor.cast %res : tensor<5xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
