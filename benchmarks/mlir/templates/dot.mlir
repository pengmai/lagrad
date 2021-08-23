func @{{prefix}}dot(%arg0: tensor<{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> tensor<f32> {
  %cst = constant dense<0.0> : tensor<f32>
  %0 = linalg.dot ins(%arg0, %arg1 : tensor<{{n}}xf32>, tensor<{{n}}xf32>) outs(%cst : tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

{% if args == [0] %}
func @{{prefix}}grad_dot_first(%arg0: tensor<{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> tensor<{{n}}xf32> {
  %f = constant @{{prefix}}dot : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<f32>
  %df = standalone.grad %f {of = [0]} : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<f32>, (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{n}}xf32>
  %res = call_indirect %df(%arg0, %arg1) : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{n}}xf32>
  return %res : tensor<{{n}}xf32>
}

{% elif args == [1] %}
func @{{prefix}}grad_dot_second(%arg0: tensor<{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> tensor<{{n}}xf32> {
  %f = constant @{{prefix}}dot : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<f32>
  %df = standalone.grad %f {of = [1]} : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<f32>, (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{n}}xf32>
  %res = call_indirect %df(%arg0, %arg1) : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{n}}xf32>
  return %res : tensor<{{n}}xf32>
}

{% elif args == [0, 1] %}
func @{{prefix}}grad_dot_both(%arg0: tensor<{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> (tensor<{{n}}xf32>, tensor<{{n}}xf32>) {
  %f = constant @{{prefix}}dot : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<f32>
  %df = standalone.grad %f {of = [0, 1]} : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<f32>, (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> (tensor<{{n}}xf32>, tensor<{{n}}xf32>)
  %res:2 = call_indirect %df(%arg0, %arg1) : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> (tensor<{{n}}xf32>, tensor<{{n}}xf32>)
  return %res#0, %res#1 : tensor<{{n}}xf32>, tensor<{{n}}xf32>
}
{% endif %}
