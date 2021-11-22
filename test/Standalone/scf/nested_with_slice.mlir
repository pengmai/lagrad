// A more complex example of differentiating for loops with nested loops and extract_slice ops.
// Still uses sum reductions instead of multiplication reductions.

#id_1d = affine_map<(d0) -> (d0)>
#reduce_1d = affine_map<(d0) -> ()>

func @nested_with_slice(%A: tensor<10x4xf64>, %B: tensor<6x4xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c10 = arith.constant 10 : index
  %zero = arith.constant 0.0 : f64
  %mt_space = arith.constant dense<0.0> : tensor<6xf64>
  %sumexp_space = arith.constant dense<0.0> : tensor<f64>
  %zerod_space = arith.constant dense<0.0> : tensor<f64>
  %final = scf.for %iv = %c0 to %c10 step %c1 iter_args(%final_iv = %zero) -> f64 {
    %main_term = scf.for %jv = %c0 to %c6 step %c1 iter_args(%mt_iter = %mt_space) -> tensor<6xf64> {
      %A_slice = tensor.extract_slice %A[%iv, 0] [1, 4] [1, 1] : tensor<10x4xf64> to tensor<4xf64>
      %B_slice = tensor.extract_slice %B[%jv, 0] [1, 4] [1, 1] : tensor<6x4xf64> to tensor<4xf64>
      %dotted = linalg.dot ins(%A_slice, %B_slice : tensor<4xf64>, tensor<4xf64>) outs(%zerod_space : tensor<f64>) -> tensor<f64>
      %dval = tensor.extract %dotted[] : tensor<f64>
      %mt_next = tensor.insert %dval into %mt_iter[%jv] : tensor<6xf64>
      scf.yield %mt_next : tensor<6xf64>
    }

    // logsumexp
    %sumexp = linalg.generic
      {
        indexing_maps = [#id_1d, #reduce_1d], iterator_types = ["reduction"]
      }
      ins(%main_term : tensor<6xf64>)
      outs(%sumexp_space : tensor<f64>) {
    ^bb0(%arg0: f64, %arg1: f64):
      %0 = math.exp %arg0 : f64
      %1 = arith.addf %0, %arg1 : f64
      linalg.yield %1 : f64
    } -> tensor<f64>
    %sumexp_v = tensor.extract %sumexp[] : tensor<f64>
    %lse = math.log %sumexp_v : f64
    %final_next = arith.addf %lse, %final_iv : f64
    scf.yield %final_next : f64
  }
  return %final : f64
}

func private @print_memref_f64(tensor<*xf64>) attributes { llvm.emit_c_interface }

func @main() {
  %A = arith.constant dense<[
    [ 1.77592084,  0.4410226 , -1.35519162,  0.52213089],
    [-0.60782562,  0.37017645, -1.29707014,  0.0302172 ],
    [-0.71204429,  0.79520425,  1.31187138,  0.87171732],
    [ 0.80214522,  0.80165727, -1.06167228, -1.36470693],
    [-1.00838663,  0.45188503,  0.29488203, -0.5173144 ],
    [-1.23645424, -0.81174622, -0.30519973, -0.78463941],
    [-0.54264342, -0.28001106,  0.6794459 ,  0.7428013 ],
    [ 1.63998002,  0.92830372, -0.08878142, -0.60710894],
    [ 0.46556958, -0.62454222, -0.00595935,  0.67031459],
    [ 0.48612281,  0.70165962, -0.45113371, -0.2207985 ]
  ]> : tensor<10x4xf64>
  %B = arith.constant dense<[
    [-0.82018689,  2.13988539, -0.30631733,  0.12993132],
    [-0.26010422, -0.08724643, -0.18346267, -0.69786579],
    [ 1.37399239, -0.11945142, -1.11526263, -0.81921764],
    [ 0.456888  , -1.38240396,  0.72714444,  1.580201  ],
    [ 0.39593011,  1.05719539,  1.10094119,  1.2460805 ],
    [-0.45650058,  0.26570091,  0.44657684,  0.57883876]
  ]> : tensor<6x4xf64>
  %f = constant @nested_with_slice : (tensor<10x4xf64>, tensor<6x4xf64>) -> f64
  %df = standalone.grad %f {of = [0]} : (tensor<10x4xf64>, tensor<6x4xf64>) -> f64, (tensor<10x4xf64>, tensor<6x4xf64>) -> tensor<10x4xf64>
  %res = call_indirect %df(%A, %B) : (tensor<10x4xf64>, tensor<6x4xf64>) -> tensor<10x4xf64>
  %U = tensor.cast %res : tensor<10x4xf64> to tensor<*xf64>
  call @print_memref_f64(%U) : (tensor<*xf64>) -> ()
  return
}
