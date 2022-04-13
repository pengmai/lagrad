//
// PRIMAL
//
func @pow(%x: f64, %cn: i64) -> f64 {
  %r_init = linalg.init_tensor [1024] : tensor<1024xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.index_cast %cn : i64 to index
  %r_t_final = scf.for %iv = %c0 to %n step %c1 iter_args(%r_t = %r_init) -> (tensor<1024xf64>) {
    %r = tensor.extract %r_t[%c0] : tensor<1024xf64>
    %r_next = arith.mulf %r, %x : f64
    %r_t_next = tensor.insert %r_next into %r_t[%c0] : tensor<1024xf64>
    scf.yield %r_t_next : tensor<1024xf64>
  }

  %r_final = tensor.extract %r_t_final[%c0] : tensor<1024xf64>
  return %r_final : f64
}


//
// ADJOINTS
//
#view = affine_map<(d0)[s0] -> (d0 + s0)>
func @grad_pow_fully_cached(%x: f64, %cn: i64) -> f64 {
  %one = arith.constant 1.0 : f64
  %r_space = linalg.init_tensor [1024] : tensor<1024xf64>
  %r_init = linalg.fill(%one, %r_space) : f64, tensor<1024xf64> -> tensor<1024xf64>
  %n = arith.index_cast %cn : i64 to index
  %r_cache = memref.alloc(%n) : memref<?x1024xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r_final = scf.for %iv = %c0 to %n step %c1 iter_args(%r_t = %r_init) -> (tensor<1024xf64>) {
    %r_view = memref.subview %r_cache[%iv, 0] [1, 1024] [1, 1] : memref<?x1024xf64> to memref<1024xf64, #view>
    %r_m = memref.buffer_cast %r_t : memref<1024xf64>
    linalg.copy(%r_m, %r_view) : memref<1024xf64>, memref<1024xf64, #view>
    %r = tensor.extract %r_t[%c0] : tensor<1024xf64>
    // memref.store %r, %r_cache[%iv] : memref<?xf64>
    %r_next = arith.mulf %r, %x : f64
    %r_t_next = tensor.insert %r_next into %r_t[%c0] : tensor<1024xf64>
    scf.yield %r_t_next : tensor<1024xf64>
  }

  %dx_init = arith.constant 0.0 : f64
  %dr_init = arith.constant 1.0 : f64
  %d_final:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%dx = %dx_init, %dr = %dr_init) -> (f64, f64) {
    %i_0 = arith.subi %n, %iv : index
    %i = arith.subi %i_0, %c1 : index
    %r = memref.load %r_cache[%i, %c0] : memref<?x1024xf64>
    %drr = arith.mulf %dr, %r : f64
    %dx_next = arith.addf %dx, %drr : f64
    %dr_next = arith.mulf %dr, %x : f64
    scf.yield %dx_next, %dr_next : f64, f64
  }
  return %d_final#0 : f64
}

func @grad_pow_smart_cached(%x: f64, %cn: i64) -> f64 {
  %one = arith.constant 1.0 : f64
  %r_space = linalg.init_tensor [1024] : tensor<1024xf64>
  %r_init = linalg.fill(%one, %r_space) : f64, tensor<1024xf64> -> tensor<1024xf64>
  %n = arith.index_cast %cn : i64 to index
  %x_cache = memref.alloc(%n) : memref<?xf64>
  %r_cache = memref.alloc(%n) : memref<?xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r_final = scf.for %iv = %c0 to %n step %c1 iter_args(%r_t = %r_init) -> (tensor<1024xf64>) {
    %r = tensor.extract %r_t[%c0] : tensor<1024xf64>
    memref.store %x, %x_cache[%iv] : memref<?xf64>
    memref.store %r, %r_cache[%iv] : memref<?xf64>
    %r_next = arith.mulf %r, %x : f64
    %r_t_next = tensor.insert %r_next into %r_t[%c0] : tensor<1024xf64>
    scf.yield %r_t_next : tensor<1024xf64>
  }

  %dx_init = arith.constant 0.0 : f64
  %dr_init = arith.constant 1.0 : f64
  %d_final:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%dx = %dx_init, %dr = %dr_init) -> (f64, f64) {
    %i_0 = arith.subi %n, %iv : index
    %i = arith.subi %i_0, %c1 : index
    %x_it = memref.load %x_cache[%i] : memref<?xf64>
    %r = memref.load %r_cache[%i] : memref<?xf64>
    %drr = arith.mulf %dr, %r : f64
    %dx_next = arith.addf %dx, %drr : f64
    %dr_next = arith.mulf %dr, %x_it : f64
    scf.yield %dx_next, %dr_next : f64, f64
  }
  return %d_final#0 : f64
}

func @grad_pow_recomputed(%x: f64, %cn: i64) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r_init = arith.constant 1.0 : f64
  %n = arith.index_cast %cn : i64 to index
  %dx_init = arith.constant 0.0 : f64
  %dr_init = arith.constant 1.0 : f64
  %d_final:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%dx = %dx_init, %dr = %dr_init) -> (f64, f64) {
    %i_0 = arith.subi %n, %iv : index
    %i = arith.subi %i_0, %c1 : index
    %x_it = scf.for %jv = %c0 to %i step %c1 iter_args(%x_j = %x) -> (f64) {
      scf.yield %x_j : f64
    }
    %r = scf.for %jv = %c0 to %i step %c1 iter_args(%r_j = %r_init) -> (f64) {
      %r_next = arith.mulf %r_j, %x : f64
      scf.yield %r_next : f64
    }

    %drr = arith.mulf %dr, %r : f64
    %dx_next = arith.addf %dx, %drr : f64
    %dr_next = arith.mulf %dr, %x_it : f64
    scf.yield %dx_next, %dr_next : f64, f64
  }
  return %d_final#0 : f64
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }
func private @print_memref_i64(tensor<*xi64>) attributes { llvm.emit_c_interface }

func @p(%val : f64) -> () {
  %space = linalg.init_tensor [] : tensor<f64>
  %t = tensor.insert %val into %space[] : tensor<f64>
  %U = tensor.cast %t : tensor<f64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}

func @pi(%val : i64) -> () {
  %space = linalg.init_tensor [] : tensor<i64>
  %t = tensor.insert %val into %space[] : tensor<i64>
  %U = tensor.cast %t : tensor<i64> to tensor<*xi64>
  call @print_memref_i64(%U) : (tensor<*xi64>) -> ()
  return
}

// func @main() {
//   %x = arith.constant 1.3 : f64
//   %n = arith.constant 4 : i64
//   %dx = call @grad_pow_fully_cached(%x, %n) : (f64, i64) -> f64
//   // call @p(%dx) : (f64) -> ()
//   // %dx2 = call @grad_pow_recomputed(%x, %n) : (f64, i64) -> f64
//   // call @p(%dx2) : (f64) -> ()
//   return
// }
