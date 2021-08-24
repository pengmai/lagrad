func @{{prefix}}matvec(%arg0: tensor<{{m}}x{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> tensor<{{m}}xf32> {
  %cst = constant dense<0.0> : tensor<{{m}}xf32>
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) outs(%cst : tensor<{{m}}xf32>) -> tensor<{{m}}xf32>
  return %0 : tensor<{{m}}xf32>
}

{% if args == [0] %}
func @{{prefix}}grad_matvec_first(%arg0: tensor<{{m}}x{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> tensor<{{m}}x{{n}}xf32> {
  %f = constant @{{prefix}}matvec : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}xf32>
  %df = standalone.grad %f {of = [0]} : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}xf32>, (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}x{{n}}xf32>
  %res = call_indirect %df(%arg0, %arg1) : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}x{{n}}xf32>
  return %res : tensor<{{m}}x{{n}}xf32>
}

{% elif args == [1] %}
func @{{prefix}}grad_matvec_second(%arg0: tensor<{{m}}x{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> tensor<{{n}}xf32> {
  %f = constant @{{prefix}}matvec : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}xf32>
  %df = standalone.grad %f {of = [1]} : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}xf32>, (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{n}}xf32>
  %res = call_indirect %df(%arg0, %arg1) : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{n}}xf32>
  return %res : tensor<{{n}}xf32>
}
{% elif args == [0, 1] %}
func @{{prefix}}grad_matvec_both(%arg0: tensor<{{m}}x{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) {
  %f = constant @{{prefix}}matvec : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}xf32>
  %df = standalone.grad %f {of = [0, 1]} : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}xf32>, (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>)
  %res:2 = call_indirect %df(%arg0, %arg1) : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>)
  return %res#0, %res#1 : tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>
}
{% endif %}
