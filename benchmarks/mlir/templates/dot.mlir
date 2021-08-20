func @dot(%arg0: tensor<{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> tensor<f32> {
  %cst = constant dense<0.0> : tensor<f32>
  %0 = linalg.dot ins(%arg0, %arg1 : tensor<{{n}}xf32>, tensor<{{n}}xf32>) outs(%cst : tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

{% if arg == [0] or arg == [1] %}
func @grad_dot(%arg0: tensor<{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> tensor<{{n}}xf32> {

{% elif arg == [0, 1] %}
func @grad_dot(%arg0: tensor<{{n}}xf32>, %arg1: tensor<{{n}}xf32>) -> (tensor<{{n}}xf32>, tensor<{{n}}xf32>) {

{% endif %}

  %f = constant @dot : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<f32>

{% if arg == [0] or arg == [1] %}
  %df = standalone.grad %f {of = {{ arg }}} : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<f32>, (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{n}}xf32>
  %res = call_indirect %df(%arg0, %arg1) : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<{{n}}xf32>
  return %res : tensor<{{n}}xf32>

{% elif arg == [0, 1] %}
  %df = standalone.grad %f {of = {{ arg }}} : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> tensor<f32>, (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> (tensor<{{n}}xf32>, tensor<{{n}}xf32>)
  %res:2 = call_indirect %df(%arg0, %arg1) : (tensor<{{n}}xf32>, tensor<{{n}}xf32>) -> (tensor<{{n}}xf32>, tensor<{{n}}xf32>)
  return %res#0, %res#1 : tensor<{{n}}xf32>, tensor<{{n}}xf32>
{% endif %}
}


func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }
func @main() {
  %cst = constant dense<1.0> : tensor<4xf32>
  %u = tensor.cast %cst : tensor<4xf32> to tensor<*xf32>
  call @print_memref_f32(%u) : (tensor<*xf32>) -> ()
  return
}
