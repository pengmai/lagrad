func @matvec(%arg0: tensor<{{m}}x{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> tensor<{{m}}xf32> {
  %cst = constant dense<0.0> : tensor<{{m}}xf32>
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) outs(%cst : tensor<{{m}}xf32>) -> tensor<{{m}}xf32>
  return %0 : tensor<{{m}}xf32>
}

{% if arg == [0] %}
func @grad_matvec(%arg0: tensor<{{m}}x{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> tensor<{{m}}x{{n}}xf32> {
{% elif arg == [1] %}
func @grad_matvec(%arg0: tensor<{{m}}x{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> tensor<{{n}}xf32> {
{% elif arg == [0, 1] %}
func @grad_matvec(%arg0: tensor<{{m}}x{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) {

{% endif %}

  %f = constant @matvec : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}xf32>
{% if arg == [0] %}
  %df = standalone.grad %f {of = [0]} : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}xf32>, (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}x{{n}}xf32>
  %res = call_indirect %df(%arg0, %arg1) : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}x{{n}}xf32>
  return %res : tensor<{{m}}x{{n}}xf32>
{% elif arg == [1] %}
  %df = standalone.grad %f {of = [1]} : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}xf32>, (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{n}}xf32>
  %res = call_indirect %df(%arg0, %arg1) : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{n}}xf32>
  return %res : tensor<{{n}}xf32>
{% elif arg == [0, 1] %}
  %df = standalone.grad %f {of = [0, 1]} : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{m}}xf32>, (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>)
  %res:2 = call_indirect %df(%arg0, %arg1) : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>) -> (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>)
  return %res#0, %res#1 : tensor<{{m}}x{{n}}xf32>, tensor<{{n}}xf32>
{% endif %}
}
