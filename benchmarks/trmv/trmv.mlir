func @mtrmv_full(%M: tensor<{{n}}x{{n}}xf64>, %x: tensor<{{n}}xf64>) -> tensor<{{n}}xf64> {
  %out = arith.constant dense<0.0> : tensor<{{n}}xf64>
  %res = linalg.matvec ins(%M, %x : tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>) outs(%out : tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  return %res : tensor<{{n}}xf64>
}

func @lagrad_trmv_full(%M: tensor<{{n}}x{{n}}xf64>, %x: tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>) {
  %f = constant @mtrmv_full : (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  %df = standalone.grad %f {of = [0, 1]} :
    (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>,
    (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>)
  %res:2 = call_indirect %df(%M, %x) : (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>)
  return %res#0, %res#1 : tensor<{{n}}x{{n}}xf64>, tensor<{{n}}xf64>
}

func @mtrmv_tri(%M: tensor<{{n}}x{{n}}xf64, "ltri">, %x: tensor<{{n}}xf64>) -> tensor<{{n}}xf64> {
  %out = arith.constant dense<0.0> : tensor<{{n}}xf64>
  %res = linalg.matvec ins(%M, %x : tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>) outs(%out : tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  return %res : tensor<{{n}}xf64>
}

func @lagrad_trmv_tri(%M: tensor<{{n}}x{{n}}xf64, "ltri">, %x: tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>) {
  %f = constant @mtrmv_tri : (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  %df = standalone.grad %f {of = [0, 1]} :
    (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>,
    (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>)
  %res:2 = call_indirect %df(%M, %x) : (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>)
  return %res#0, %res#1 : tensor<{{n}}x{{n}}xf64, "ltri">, tensor<{{n}}xf64>
}


func @mtrmv_packed(%M: tensor<{{n}}x{{n}}xf64, "pltri">, %x: tensor<{{n}}xf64>) -> tensor<{{n}}xf64> {
  %out = arith.constant dense<0.0> : tensor<{{n}}xf64>
  %res = linalg.generic
    {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]
    }
    ins(%M, %x : tensor<{{n}}x{{n}}xf64, "pltri">, tensor<{{n}}xf64>)
    outs(%out : tensor<{{n}}xf64>) {
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %0 = arith.mulf %arg0, %arg1 : f64
    %1 = arith.addf %0, %arg2 : f64
    linalg.yield %1 : f64
  } -> tensor<{{n}}xf64>
  return %res : tensor<{{n}}xf64>
}

func @lagrad_trmv_packed(%M: tensor<{{n}}x{{n}}xf64, "pltri">, %x: tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64, "pltri">, tensor<{{n}}xf64>) {
  %f = constant @mtrmv_packed : (tensor<{{n}}x{{n}}xf64, "pltri">, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>
  %df = standalone.grad %f {of = [0, 1]} :
    (tensor<{{n}}x{{n}}xf64, "pltri">, tensor<{{n}}xf64>) -> tensor<{{n}}xf64>,
    (tensor<{{n}}x{{n}}xf64, "pltri">, tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64, "pltri">, tensor<{{n}}xf64>)
  %res:2 = call_indirect %df(%M, %x) : (tensor<{{n}}x{{n}}xf64, "pltri">, tensor<{{n}}xf64>) -> (tensor<{{n}}x{{n}}xf64, "pltri">, tensor<{{n}}xf64>)
  return %res#0, %res#1 : tensor<{{n}}x{{n}}xf64, "pltri">, tensor<{{n}}xf64>
}
