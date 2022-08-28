func @emlir_compute_zach_weight_error(%w: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %0 = arith.mulf %w, %w : f64
  %1 = arith.subf %one, %0 : f64
  return %1 : f64
}

func @enzyme_compute_w_error(%w: f64) -> f64 {
  %f = constant @emlir_compute_zach_weight_error : (f64) -> f64
  %df = standalone.diff %f : (f64) -> f64, (f64) -> f64
  %dw = call_indirect %df(%w) : (f64) -> f64
  return %dw : f64
}
