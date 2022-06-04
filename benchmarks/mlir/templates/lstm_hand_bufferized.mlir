#oned = affine_map<(d0) -> (d0)>
#twod = affine_map<(d0, d1) -> (d0, d1)>
#view = affine_map<(d0)[s0] -> (d0 + s0)>
#view2 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>

func private @print_memref_f64(memref<*xf64>) attributes { llvm.emit_c_interface }

func private @msigmoid_hb(%x: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %nx = arith.negf %x : f64
  %exp = math.exp %nx : f64
  %denom = arith.addf %one, %exp : f64
  %frac = arith.divf %one, %denom : f64
  return %frac : f64
}

func private @grad_sigmoid_hb(%x: f64, %g: f64) -> f64 {
  %cst = arith.constant 1.000000e+00 : f64
  %0 = arith.negf %x : f64
  %1 = math.exp %0 : f64
  %2 = arith.addf %cst, %1 : f64
  %s = arith.divf %cst, %2 : f64
  %4 = arith.mulf %g, %s : f64
  %5 = arith.subf %cst, %s : f64
  %6 = arith.mulf %4, %5 : f64
  return %6 : f64
}

func private @grad_tanh_hb(%x: f64, %g: f64) -> f64 {
  %exp = math.exp %x : f64
  %nx = arith.negf %x : f64
  %negexp = math.exp %nx : f64
  %numerator = arith.addf %exp, %negexp : f64
  %half = arith.constant 0.5 : f64
  %cosh = arith.mulf %numerator, %half : f64
  %cosh2 = arith.mulf %cosh, %cosh : f64
  %dx = arith.divf %g, %cosh2 : f64
  return %dx : f64
}

func private @mlogsumexp_hb(%t: memref<{{b}}xf64>) -> f64 {
  %zero = arith.constant 0.0 : f64
  %lse = memref.alloca() : memref<f64>
  memref.store %zero, %lse[] : memref<f64>

  linalg.generic
    {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    }
    ins(%t : memref<{{b}}xf64>)
    outs(%lse : memref<f64>) {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = math.exp %arg0 : f64
    %1 = arith.addf %0, %arg1 : f64
    linalg.yield %1 : f64
  }
  %lse_v = memref.load %lse[] : memref<f64>
  %two = arith.constant 2.0 : f64
  %lse_2 = arith.addf %lse_v, %two : f64
  %lse_l = math.log %lse_2 : f64
  return %lse_l : f64
}

func @lstm_model_hb(
  %weight: memref<4x{{b}}xf64, #view2>,
  %bias: memref<4x{{b}}xf64, #view2>,
  %hidden: memref<{{b}}xf64, #view>,
  %cell: memref<{{b}}xf64, #view>,
  %input: memref<{{b}}xf64>
) {
  %gates   = memref.alloc() : memref<4x{{b}}xf64>
  %forget  = memref.subview %gates[0, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  %ingate  = memref.subview %gates[1, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  %outgate = memref.subview %gates[2, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  %change  = memref.subview %gates[3, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>

  %fweight = memref.subview %weight[0, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %fbias   = memref.subview %bias  [0, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %iweight = memref.subview %weight[1, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %ibias   = memref.subview %bias  [1, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %oweight = memref.subview %weight[2, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %obias   = memref.subview %bias  [2, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %cweight = memref.subview %weight[3, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %cbias   = memref.subview %bias  [3, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%input, %fweight, %fbias : memref<{{b}}xf64>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%forget : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    %2 = call @msigmoid_hb(%1) : (f64) -> f64
    linalg.yield %2 : f64
  }
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%hidden, %iweight, %ibias : memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%ingate : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    %2 = call @msigmoid_hb(%1) : (f64) -> f64
    linalg.yield %2 : f64
  }
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%input, %oweight, %obias : memref<{{b}}xf64>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%outgate : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    %2 = call @msigmoid_hb(%1) : (f64) -> f64
    linalg.yield %2 : f64
  }
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%hidden, %cweight, %cbias : memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%change : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    %2 = math.tanh %1 : f64
    linalg.yield %2 : f64
  }

  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%cell, %forget, %ingate, %change : memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%cell : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.mulf %arg2, %arg3 : f64
    %2 = arith.addf %0, %1 : f64
    linalg.yield %2 : f64
  }

  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%outgate, %cell : memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%hidden : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = math.tanh %arg1 : f64
    %1 = arith.mulf %arg0, %0 : f64
    linalg.yield %1 : f64
  }

  // Optionally dealloc %gates
  return
}

func @grad_lstm_model_hb(
  %weight: memref<4x{{b}}xf64, #view2>,
  %dweight: memref<4x{{b}}xf64>,
  %bias: memref<4x{{b}}xf64, #view2>,
  %dbias: memref<4x{{b}}xf64>,
  %hidden: memref<{{b}}xf64, #view>,
  %dhidden: memref<{{b}}xf64>,
  %cell: memref<{{b}}xf64, #view>,
  %dcell: memref<{{b}}xf64>,
  %input: memref<{{b}}xf64>,
  %dinput: memref<{{b}}xf64>
) {
  %gates   = memref.alloc() : memref<4x{{b}}xf64>
  %forget  = memref.subview %gates[0, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  %ingate  = memref.subview %gates[1, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  %outgate = memref.subview %gates[2, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  %change  = memref.subview %gates[3, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>

  %fweight = memref.subview %weight[0, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %fbias   = memref.subview %bias  [0, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %iweight = memref.subview %weight[1, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %ibias   = memref.subview %bias  [1, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %oweight = memref.subview %weight[2, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %obias   = memref.subview %bias  [2, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %cweight = memref.subview %weight[3, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  %cbias   = memref.subview %bias  [3, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64, #view2> to memref<{{b}}xf64, #view>
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%input, %fweight, %fbias : memref<{{b}}xf64>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%forget : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    linalg.yield %1 : f64
  }
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%hidden, %iweight, %ibias : memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%ingate : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    linalg.yield %1 : f64
  }
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%input, %oweight, %obias : memref<{{b}}xf64>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%outgate : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    linalg.yield %1 : f64
  }
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%hidden, %cweight, %cbias : memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%change : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    linalg.yield %1 : f64
  }

  %gates_old = memref.alloc() : memref<4x{{b}}xf64>
  linalg.copy(%gates, %gates_old) : memref<4x{{b}}xf64>, memref<4x{{b}}xf64>

  %sigmoidgates = memref.subview %gates[0, 0] [3, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<3x{{b}}xf64>
  linalg.generic
    { indexing_maps = [#twod], iterator_types = ["parallel", "parallel"] }
    outs(%sigmoidgates : memref<3x{{b}}xf64>) {
  ^bb0(%arg0: f64):
    %0 = call @msigmoid_hb(%arg0) : (f64) -> f64
    linalg.yield %0 : f64
  }
  linalg.generic
    { indexing_maps = [#oned], iterator_types = ["parallel"] }
    outs(%change : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64):
    %0 = math.tanh %arg0 : f64
    linalg.yield %0 : f64
  }

  %cell_old = memref.alloc() : memref<{{b}}xf64>
  linalg.copy(%cell, %cell_old) : memref<{{b}}xf64, #view>, memref<{{b}}xf64>

  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%cell, %forget, %ingate, %change : memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%cell : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.mulf %arg2, %arg3 : f64
    %2 = arith.addf %0, %1 : f64
    linalg.yield %2 : f64
  }

  // Removed the hidden computation because it's not needed and the old hidden value is required
  %doutgate = memref.alloc() : memref<{{b}}xf64>
  %outgate_presig = memref.subview %gates_old[2, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%dhidden, %cell, %outgate_presig : memref<{{b}}xf64>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%doutgate : memref<{{b}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = math.tanh %arg1 : f64
    %1 = arith.mulf %arg0, %0 : f64
    %2 = call @grad_sigmoid_hb(%arg2, %1) : (f64, f64) -> f64
    linalg.yield %2 : f64
  }

  %dcell_new = memref.alloc() : memref<{{b}}xf64>
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%dcell, %cell, %outgate, %dhidden : memref<{{b}}xf64>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>, memref<{{b}}xf64>)
    outs(%dcell_new : memref<{{b}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64):
    %g = arith.mulf %arg2, %arg3 : f64
    %gtanh = call @grad_tanh_hb(%arg1, %g) : (f64, f64) -> f64
    %0 = arith.addf %arg0, %gtanh : f64
    linalg.yield %0 : f64
  }
  %dforget = memref.alloc() : memref<{{b}}xf64>
  %forget_presig = memref.subview %gates_old[0, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%cell_old, %dcell_new, %forget_presig : memref<{{b}}xf64>, memref<{{b}}xf64>, memref<{{b}}xf64, #view>)
    outs(%dforget : memref<{{b}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = call @grad_sigmoid_hb(%arg2, %0) : (f64, f64) -> f64
    linalg.yield %1 : f64
  }

  %dingate = memref.alloc() : memref<{{b}}xf64>
  %ingate_presig = memref.subview %gates_old[1, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%dcell_new, %change, %ingate_presig : memref<{{b}}xf64>, memref<{{b}}xf64, #view>, memref<{{b}}xf64, #view>)
    outs(%dingate : memref<{{b}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = call @grad_sigmoid_hb(%arg2, %0) : (f64, f64) -> f64
    linalg.yield %1 : f64
  }

  %dchange = memref.alloc() : memref<{{b}}xf64>
  %change_pretanh = memref.subview %gates_old[3, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%ingate, %dcell_new, %change_pretanh : memref<{{b}}xf64, #view>, memref<{{b}}xf64>, memref<{{b}}xf64, #view>)
    outs(%dchange : memref<{{b}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = call @grad_tanh_hb(%arg2, %0) : (f64, f64) -> f64
    linalg.yield %1 : f64
  }

  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%dcell_new, %forget : memref<{{b}}xf64>, memref<{{b}}xf64, #view>)
    outs(%dcell : memref<{{b}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  }

  %dbias0 = memref.subview %dbias[0, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  linalg.copy(%dforget, %dbias0) : memref<{{b}}xf64>, memref<{{b}}xf64, #view>
  %dbias1 = memref.subview %dbias[1, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  linalg.copy(%dingate, %dbias1) : memref<{{b}}xf64>, memref<{{b}}xf64, #view>
  %dbias2 = memref.subview %dbias[2, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  linalg.copy(%doutgate, %dbias2) : memref<{{b}}xf64>, memref<{{b}}xf64, #view>
  %dbias3 = memref.subview %dbias[3, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  linalg.copy(%dchange, %dbias3) : memref<{{b}}xf64>, memref<{{b}}xf64, #view>

  %dweight0 = memref.subview %dweight[0, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%input, %dforget : memref<{{b}}xf64>, memref<{{b}}xf64>)
    outs(%dweight0 : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  }
  %dweight1 = memref.subview %dweight[1, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%hidden, %dingate : memref<{{b}}xf64, #view>, memref<{{b}}xf64>)
    outs(%dweight1 : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  }
  %dweight2 = memref.subview %dweight[2, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%input, %doutgate : memref<{{b}}xf64>, memref<{{b}}xf64>)
    outs(%dweight2 : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  }
  %dweight3 = memref.subview %dweight[3, 0] [1, {{b}}] [1, 1] : memref<4x{{b}}xf64> to memref<{{b}}xf64, #view>
  linalg.generic
    { indexing_maps = [#oned, #oned, #oned], iterator_types = ["parallel"] }
    ins(%hidden, %dchange : memref<{{b}}xf64, #view>, memref<{{b}}xf64>)
    outs(%dweight3 : memref<{{b}}xf64, #view>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    linalg.yield %0 : f64
  }

  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%dforget, %fweight, %doutgate, %oweight : memref<{{b}}xf64>, memref<{{b}}xf64, #view>, memref<{{b}}xf64>, memref<{{b}}xf64, #view>)
    outs(%dinput : memref<{{b}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.mulf %arg2, %arg3 : f64
    %2 = arith.addf %0, %1 : f64
    linalg.yield %2 : f64
  }

  linalg.generic
    {
      indexing_maps = [#oned, #oned, #oned, #oned, #oned],
      iterator_types = ["parallel"]
    }
    ins(%dingate, %iweight, %dchange, %cweight : memref<{{b}}xf64>, memref<{{b}}xf64, #view>, memref<{{b}}xf64>, memref<{{b}}xf64, #view>)
    outs(%dhidden : memref<{{b}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.mulf %arg2, %arg3 : f64
    %2 = arith.addf %0, %1 : f64
    linalg.yield %2 : f64
  }

  return
}
