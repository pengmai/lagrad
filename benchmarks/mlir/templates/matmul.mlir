func @{{prefix}}matmul(%arg0: tensor<{{m}}x{{n}}xf32>, %arg1: tensor<{{n}}x{{k}}xf32>) -> tensor<{{m}}x{{k}}xf32> {
  %cst = constant dense<0.0> : tensor<{{m}}x{{k}}xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) outs(%cst : tensor<{{m}}x{{k}}xf32>) -> tensor<{{m}}x{{k}}xf32>
  return %0 : tensor<{{m}}x{{k}}xf32>
}

{% if args == [0] %}
func @{{prefix}}grad_matmul_first(%arg0: tensor<{{m}}x{{n}}xf32>, %arg1: tensor<{{n}}x{{k}}xf32>) -> tensor<{{m}}x{{n}}xf32> {
  %f = constant @{{prefix}}matmul : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) -> tensor<{{m}}x{{k}}xf32>
  %df = standalone.grad %f {of = [0]} : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) -> tensor<{{m}}x{{k}}xf32>, (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) -> tensor<{{m}}x{{n}}xf32>
  %res = call_indirect %df(%arg0, %arg1) : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) -> tensor<{{m}}x{{n}}xf32>
  return %res : tensor<{{m}}x{{n}}xf32>
}

{% elif args == [1] %}
func @{{prefix}}grad_matmul_second(%arg0: tensor<{{m}}x{{n}}xf32>, %arg1: tensor<{{n}}x{{k}}xf32>) -> tensor<{{n}}x{{k}}xf32> {
  %f = constant @{{prefix}}matmul : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) -> tensor<{{m}}x{{k}}xf32>
  %df = standalone.grad %f {of = [1]} : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) -> tensor<{{m}}x{{k}}xf32>, (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) -> tensor<{{n}}x{{k}}xf32>
  %res = call_indirect %df(%arg0, %arg1) : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) -> tensor<{{n}}x{{k}}xf32>
  return %res : tensor<{{n}}x{{k}}xf32>
}

{% elif args == [0, 1] %}
func @{{prefix}}grad_matmul_both(%arg0: tensor<{{m}}x{{n}}xf32>, %arg1: tensor<{{n}}x{{k}}xf32>) -> (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) {
  %f = constant @{{prefix}}matmul : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) -> tensor<{{m}}x{{k}}xf32>
  %df = standalone.grad %f {of = [0, 1]} : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) -> tensor<{{m}}x{{k}}xf32>, (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) -> (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>)
  %res:2 = call_indirect %df(%arg0, %arg1) : (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>) -> (tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>)
  return %res#0, %res#1 : tensor<{{m}}x{{n}}xf32>, tensor<{{n}}x{{k}}xf32>
}
{% endif %}
