// Minimal working example of a dot product.
func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

//declare double @__enzyme_autodiff(i8*, double) #1
// llvm.func @__enzyme_autodiff(!llvm.ptr<i8>, ...) -> f32

func @print_0d(%arg : memref<f32>) {
  %U = memref_cast %arg :  memref<f32> to memref<*xf32>
  call @print_memref_f32(%U) : (memref<*xf32>) -> ()
  return
}

func @mul2(%arg : f32) -> f32 {
  %two = constant 2.0 : f32
  %res = mulf %arg, %two : f32
  return %res : f32
}

func @main() {
  %f = constant 1.3 : f32
  // %A = alloc() : memref<16xf32>
  // %B = alloc() : memref<16xf32>
  // %res = alloc() : memref<f32>
  // linalg.fill(%A, %f) : memref<16xf32>, f32
  // linalg.fill(%B, %f) : memref<16xf32>, f32

  // Take the dot product. Note the special syntax.
  // linalg.dot ins(%A, %B : memref<16xf32>, memref<16xf32>) outs(%res : memref<f32>)
  // %val = llvm.call @dot(%A, %B) : (!llvm.struct<(i64, ptr<i8>)>, !llvm.struct<(i64, ptr<i8>)>) -> (f32)
  // store %val, %res[] : memref<f32>
  // %loaded = load %res[] : memref<f32>
  %addfunc = constant @mul2 : (f32) -> f32
  // store %val, %res[] : memref<f32>

  %blah = standalone.diff %addfunc : (f32) -> f32
  %myres = call_indirect %blah(%f) : (f32) -> f32
  // %myres = call_indirect %addfunc(%f) : (f32) -> f32

  // %dres = llvm.call @__enzyme_autodiff()

  // Print the result.
  // call @print_0d(%res) : (memref<f32>) -> ()
  // dealloc %A : memref<16xf32>
  // dealloc %B : memref<16xf32>
  return
}
