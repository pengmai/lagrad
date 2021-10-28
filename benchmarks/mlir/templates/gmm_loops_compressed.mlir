module  {
  memref.global "private" constant @__constant_xf64_1 : memref<f64> = dense<1.000000e+00>
  memref.global "private" constant @__constant_200x8128xf64 : memref<200x8128xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_200x128xf64 : memref<200x128xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1000x200x128xf64 : memref<1000x200x128xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1000x200xf64 : memref<1000x200xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1000xf64 : memref<1000xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_200xf64 : memref<200xf64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf64_0 : memref<f64> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<1.000000e+03>
  func private @print_memref_f64(memref<*xf64>) attributes {llvm.emit_c_interface}
  func @__grad_gmm_objective(%arg0: memref<200xf64>, %arg1: memref<200x128xf64>, %arg2: memref<200x128xf64>, %arg3: memref<200x8128xf64>, %arg4: memref<1000x128xf64>, %arg5: f64, %arg6: i64) -> (memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x8128xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e-01 : f64
    %c25 = arith.constant 25 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %0 = memref.get_global @__constant_200x128xf64 : memref<200x128xf64>
    %1 = memref.get_global @__constant_1000x200x128xf64 : memref<1000x200x128xf64>
    %2 = memref.get_global @__constant_200x8128xf64 : memref<200x8128xf64>
    %3 = memref.get_global @__constant_1000x200xf64 : memref<1000x200xf64>
    %4 = memref.get_global @__constant_200xf64 : memref<200xf64>
    %5 = memref.get_global @__constant_1000xf64 : memref<1000xf64>
    %6 = memref.get_global @__constant_xf64_0 : memref<f64>
    %7 = memref.get_global @__constant_xf64_1 : memref<f64>
    %8 = memref.get_global @__constant_xf64 : memref<f64>
    %9 = memref.alloc() : memref<200x128xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 {
        %69 = memref.load %arg2[%arg7, %arg8] : memref<200x128xf64>
        %70 = math.exp %69 : f64
        memref.store %70, %9[%arg7, %arg8] : memref<200x128xf64>
      }
    }
    %10 = memref.alloc() : memref<200xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %4[%arg7] : memref<200xf64>
      memref.store %69, %10[%arg7] : memref<200xf64>
    }
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 {
        %69 = memref.load %arg2[%arg7, %arg8] : memref<200x128xf64>
        %70 = memref.load %10[%arg7] : memref<200xf64>
        %71 = arith.addf %69, %70 : f64
        memref.store %71, %10[%arg7] : memref<200xf64>
      }
    }
    %11 = memref.alloc() : memref<1000x200x128xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %arg4[%arg7, %arg9] : memref<1000x128xf64>
          %70 = memref.load %arg1[%arg8, %arg9] : memref<200x128xf64>
          %71 = arith.subf %69, %70 : f64
          memref.store %71, %11[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
        }
      }
    }
    %12 = memref.alloc() : memref<1000x200x128xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %1[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          memref.store %69, %12[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
        }
      }
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %zero = arith.constant 0 : index
        scf.for %arg9 = %c0 to %c10 step %c1 iter_args(%idx = %zero) -> index {
          %id_next = scf.for %arg10 = %c0 to %arg9 step %c1 iter_args(%idy = %idx) -> index{
            %69 = memref.load %arg3[%arg8, %idy] : memref<200x8128xf64>
            %70 = memref.load %11[%arg7, %arg8, %arg10] : memref<1000x200x128xf64>
            %71 = memref.load %12[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
            %72 = arith.mulf %69, %70 : f64
            %73 = arith.addf %72, %71 : f64
            memref.store %73, %12[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
            %idz = arith.addi %c1, %idy : index
            scf.yield %idz : index
          }
          scf.yield %id_next : index
        }
      }
    }
    %13 = memref.alloc() : memref<1000x200x128xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %9[%arg8, %arg9] : memref<200x128xf64>
          %70 = memref.load %11[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %71 = memref.load %12[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %72 = arith.mulf %69, %70 : f64
          %73 = arith.addf %72, %71 : f64
          %74 = arith.mulf %73, %73 : f64
          memref.store %74, %13[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
        }
      }
    }
    %14 = memref.alloc() : memref<1000x200xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %3[%arg7, %arg8] : memref<1000x200xf64>
        memref.store %69, %14[%arg7, %arg8] : memref<1000x200xf64>
      }
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %13[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %70 = memref.load %14[%arg7, %arg8] : memref<1000x200xf64>
          %71 = arith.addf %69, %70 : f64
          memref.store %71, %14[%arg7, %arg8] : memref<1000x200xf64>
        }
      }
    }
    %15 = memref.alloc() : memref<200xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %arg0[%arg7] : memref<200xf64>
      %70 = memref.load %10[%arg7] : memref<200xf64>
      %71 = arith.addf %69, %70 : f64
      memref.store %71, %15[%arg7] : memref<200xf64>
    }
    %16 = memref.alloc() : memref<1000x200xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %15[%arg8] : memref<200xf64>
        %70 = memref.load %14[%arg7, %arg8] : memref<1000x200xf64>
        %71 = arith.mulf %70, %cst_0 : f64
        %72 = arith.subf %69, %71 : f64
        memref.store %72, %16[%arg7, %arg8] : memref<1000x200xf64>
      }
    }
    %17 = memref.alloc() : memref<1000xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      %69 = memref.load %5[%arg7] : memref<1000xf64>
      memref.store %69, %17[%arg7] : memref<1000xf64>
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %16[%arg7, %arg8] : memref<1000x200xf64>
        %70 = memref.load %17[%arg7] : memref<1000xf64>
        %71 = arith.cmpf ogt, %69, %70 : f64
        %72 = select %71, %69, %70 : f64
        memref.store %72, %17[%arg7] : memref<1000xf64>
      }
    }
    %18 = memref.alloc() : memref<1000xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      %69 = memref.load %5[%arg7] : memref<1000xf64>
      memref.store %69, %18[%arg7] : memref<1000xf64>
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %16[%arg7, %arg8] : memref<1000x200xf64>
        %70 = memref.load %17[%arg7] : memref<1000xf64>
        %71 = memref.load %18[%arg7] : memref<1000xf64>
        %72 = arith.subf %69, %70 : f64
        %73 = math.exp %72 : f64
        %74 = arith.addf %73, %71 : f64
        memref.store %74, %18[%arg7] : memref<1000xf64>
      }
    }
    %19 = memref.alloc() : memref<1000xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      %69 = memref.load %18[%arg7] : memref<1000xf64>
      %70 = math.log %69 : f64
      memref.store %70, %19[%arg7] : memref<1000xf64>
    }
    %20 = memref.alloc() : memref<1000xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      %69 = memref.load %19[%arg7] : memref<1000xf64>
      %70 = memref.load %17[%arg7] : memref<1000xf64>
      %71 = arith.addf %69, %70 : f64
      memref.store %71, %20[%arg7] : memref<1000xf64>
    }
    memref.dealloc %19 : memref<1000xf64>
    %21 = memref.alloc() : memref<200xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %4[%arg7] : memref<200xf64>
      memref.store %69, %21[%arg7] : memref<200xf64>
    }
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 {
        %69 = memref.load %9[%arg7, %arg8] : memref<200x128xf64>
        %70 = memref.load %21[%arg7] : memref<200xf64>
        %71 = arith.mulf %69, %69 : f64
        %72 = arith.addf %71, %70 : f64
        memref.store %72, %21[%arg7] : memref<200xf64>
      }
    }
    %22 = memref.alloc() : memref<200xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %4[%arg7] : memref<200xf64>
      memref.store %69, %22[%arg7] : memref<200xf64>
    }
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %zero = arith.constant 0 : index
      scf.for %arg8 = %c0 to %c10 step %c1 iter_args(%idx = %zero) -> index {
        %id_next = scf.for %arg9 = %c0 to %arg8 step %c1 iter_args(%idy = %idx) -> index {
          %69 = memref.load %arg3[%arg7, %idy] : memref<200x8128xf64>
          %70 = memref.load %22[%arg7] : memref<200xf64>
          %71 = arith.mulf %69, %69 : f64
          %72 = arith.addf %71, %70 : f64
          memref.store %72, %22[%arg7] : memref<200xf64>
          %idz = arith.addi %c1, %idy : index
          scf.yield %idz : index
        }
        scf.yield %id_next : index
      }
    }
    %23 = memref.alloc() : memref<200xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %21[%arg7] : memref<200xf64>
      %70 = memref.load %22[%arg7] : memref<200xf64>
      %71 = arith.addf %69, %70 : f64
      memref.store %71, %23[%arg7] : memref<200xf64>
    }
    memref.dealloc %22 : memref<200xf64>
    memref.dealloc %21 : memref<200xf64>
    %24 = memref.alloc() : memref<f64>
    %25 = memref.load %6[] : memref<f64>
    memref.store %25, %24[] : memref<f64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %arg0[%arg7] : memref<200xf64>
      %70 = memref.load %24[] : memref<f64>
      %71 = math.exp %69 : f64
      %72 = arith.addf %71, %70 : f64
      memref.store %72, %24[] : memref<f64>
    }
    %26 = memref.alloc() : memref<f64>
    %27 = memref.load %7[] : memref<f64>
    %28 = arith.negf %27 : f64
    memref.store %28, %26[] : memref<f64>
    %29 = memref.alloc() : memref<f64>
    %30 = memref.load %26[] : memref<f64>
    %31 = memref.load %8[] : memref<f64>
    %32 = arith.mulf %30, %31 : f64
    memref.store %32, %29[] : memref<f64>
    memref.dealloc %26 : memref<f64>
    %33 = memref.alloc() : memref<f64>
    %34 = memref.load %29[] : memref<f64>
    %35 = memref.load %24[] : memref<f64>
    %36 = arith.divf %34, %35 : f64
    memref.store %36, %33[] : memref<f64>
    memref.dealloc %29 : memref<f64>
    memref.dealloc %24 : memref<f64>
    %37 = memref.alloc() : memref<200xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %4[%arg7] : memref<200xf64>
      memref.store %69, %37[%arg7] : memref<200xf64>
    }
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %arg0[%arg7] : memref<200xf64>
      %70 = memref.load %33[] : memref<f64>
      %71 = memref.load %37[%arg7] : memref<200xf64>
      %72 = math.exp %69 : f64
      %73 = arith.mulf %70, %72 : f64
      %74 = arith.addf %73, %71 : f64
      memref.store %74, %37[%arg7] : memref<200xf64>
    }
    memref.dealloc %33 : memref<f64>
    %38 = memref.alloc() : memref<f64>
    %39 = memref.load %7[] : memref<f64>
    %40 = arith.negf %39 : f64
    memref.store %40, %38[] : memref<f64>
    %41 = memref.alloc() : memref<200xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %4[%arg7] : memref<200xf64>
      memref.store %69, %41[%arg7] : memref<200xf64>
    }
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %38[] : memref<f64>
      %70 = memref.load %41[%arg7] : memref<200xf64>
      %71 = arith.sitofp %arg6 : i64 to f64
      %72 = arith.mulf %69, %71 : f64
      %73 = arith.addf %72, %70 : f64
      memref.store %73, %41[%arg7] : memref<200xf64>
    }
    memref.dealloc %38 : memref<f64>
    memref.dealloc %10 : memref<200xf64>
    %42 = memref.alloc() : memref<200xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %4[%arg7] : memref<200xf64>
      memref.store %69, %42[%arg7] : memref<200xf64>
    }
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %7[] : memref<f64>
      %70 = memref.load %42[%arg7] : memref<200xf64>
      %71 = arith.mulf %cst_0, %arg5 : f64
      %72 = arith.mulf %71, %arg5 : f64
      %73 = arith.mulf %69, %72 : f64
      %74 = arith.addf %73, %70 : f64
      memref.store %74, %42[%arg7] : memref<200xf64>
    }
    memref.dealloc %23 : memref<200xf64>
    %43 = memref.alloc() : memref<200x8128xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 iter_args(%idx = %c0) -> index {
        %id_next = scf.for %arg9 = %c0 to %arg8 step %c1 iter_args(%idy = %idx) -> index {
          %69 = memref.load %2[%arg7, %idy] : memref<200x8128xf64>
          memref.store %69, %43[%arg7, %idy] : memref<200x8128xf64>
          %idz = arith.addi %c1, %idy : index
          scf.yield %idz : index
        }
        scf.yield %id_next : index
      }
    }
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 iter_args(%idx = %c0) -> index {
        %id_next = scf.for %arg9 = %c0 to %arg8 step %c1 iter_args(%idy = %idx) -> index {
          %69 = memref.load %arg3[%arg7, %idy] : memref<200x8128xf64>
          %70 = memref.load %42[%arg7] : memref<200xf64>
          %71 = memref.load %43[%arg7, %idy] : memref<200x8128xf64>
          %72 = arith.mulf %70, %69 : f64
          %73 = arith.mulf %70, %69 : f64
          %74 = arith.addf %72, %73 : f64
          %75 = arith.addf %74, %71 : f64
          memref.store %75, %43[%arg7, %idy] : memref<200x8128xf64>
          %idz = arith.addi %idy, %c1 : index
          scf.yield %idz : index
        }
        scf.yield %id_next : index
      }
    }
    %44 = memref.alloc() : memref<200x128xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 {
        %69 = memref.load %0[%arg7, %arg8] : memref<200x128xf64>
        memref.store %69, %44[%arg7, %arg8] : memref<200x128xf64>
      }
    }
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 {
        %69 = memref.load %9[%arg7, %arg8] : memref<200x128xf64>
        %70 = memref.load %42[%arg7] : memref<200xf64>
        %71 = memref.load %44[%arg7, %arg8] : memref<200x128xf64>
        %72 = arith.mulf %70, %69 : f64
        %73 = arith.mulf %70, %69 : f64
        %74 = arith.addf %72, %73 : f64
        %75 = arith.addf %74, %71 : f64
        memref.store %75, %44[%arg7, %arg8] : memref<200x128xf64>
      }
    }
    memref.dealloc %42 : memref<200xf64>
    %45 = memref.alloc() : memref<1000xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      %69 = memref.load %5[%arg7] : memref<1000xf64>
      memref.store %69, %45[%arg7] : memref<1000xf64>
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      %69 = memref.load %7[] : memref<f64>
      %70 = memref.load %45[%arg7] : memref<1000xf64>
      %71 = arith.addf %69, %70 : f64
      memref.store %71, %45[%arg7] : memref<1000xf64>
    }
    memref.dealloc %20 : memref<1000xf64>
    %46 = memref.alloc() : memref<1000xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      %69 = memref.load %45[%arg7] : memref<1000xf64>
      %70 = memref.load %18[%arg7] : memref<1000xf64>
      %71 = arith.divf %69, %70 : f64
      memref.store %71, %46[%arg7] : memref<1000xf64>
    }
    memref.dealloc %18 : memref<1000xf64>
    %47 = memref.alloc() : memref<1000x200xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %3[%arg7, %arg8] : memref<1000x200xf64>
        memref.store %69, %47[%arg7, %arg8] : memref<1000x200xf64>
      }
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %16[%arg7, %arg8] : memref<1000x200xf64>
        %70 = memref.load %17[%arg7] : memref<1000xf64>
        %71 = memref.load %46[%arg7] : memref<1000xf64>
        %72 = memref.load %47[%arg7, %arg8] : memref<1000x200xf64>
        %73 = arith.subf %69, %70 : f64
        %74 = math.exp %73 : f64
        %75 = arith.mulf %71, %74 : f64
        %76 = arith.addf %75, %72 : f64
        memref.store %76, %47[%arg7, %arg8] : memref<1000x200xf64>
      }
    }
    %48 = memref.alloc() : memref<1000xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      %69 = memref.load %5[%arg7] : memref<1000xf64>
      memref.store %69, %48[%arg7] : memref<1000xf64>
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %16[%arg7, %arg8] : memref<1000x200xf64>
        %70 = memref.load %17[%arg7] : memref<1000xf64>
        %71 = memref.load %46[%arg7] : memref<1000xf64>
        %72 = memref.load %48[%arg7] : memref<1000xf64>
        %73 = arith.subf %69, %70 : f64
        %74 = math.exp %73 : f64
        %75 = arith.mulf %71, %74 : f64
        %76 = arith.negf %75 : f64
        %77 = arith.addf %76, %72 : f64
        memref.store %77, %48[%arg7] : memref<1000xf64>
      }
    }
    memref.dealloc %46 : memref<1000xf64>
    memref.dealloc %17 : memref<1000xf64>
    %49 = memref.alloc() : memref<1000xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      %69 = memref.load %45[%arg7] : memref<1000xf64>
      %70 = memref.load %48[%arg7] : memref<1000xf64>
      %71 = arith.addf %69, %70 : f64
      memref.store %71, %49[%arg7] : memref<1000xf64>
    }
    memref.dealloc %48 : memref<1000xf64>
    memref.dealloc %45 : memref<1000xf64>
    %50 = memref.alloc() : memref<1000x200xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %3[%arg7, %arg8] : memref<1000x200xf64>
        memref.store %69, %50[%arg7, %arg8] : memref<1000x200xf64>
      }
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %16[%arg7, %arg8] : memref<1000x200xf64>
        %70 = memref.load %49[%arg7] : memref<1000xf64>
        %71 = memref.load %50[%arg7, %arg8] : memref<1000x200xf64>
        %72 = arith.cmpf ogt, %69, %71 : f64
        %73 = select %72, %70, %cst : f64
        %74 = arith.addf %73, %71 : f64
        memref.store %74, %50[%arg7, %arg8] : memref<1000x200xf64>
      }
    }
    memref.dealloc %49 : memref<1000xf64>
    memref.dealloc %16 : memref<1000x200xf64>
    %51 = memref.alloc() : memref<1000x200xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %47[%arg7, %arg8] : memref<1000x200xf64>
        %70 = memref.load %50[%arg7, %arg8] : memref<1000x200xf64>
        %71 = arith.addf %69, %70 : f64
        memref.store %71, %51[%arg7, %arg8] : memref<1000x200xf64>
      }
    }
    memref.dealloc %50 : memref<1000x200xf64>
    memref.dealloc %47 : memref<1000x200xf64>
    %52 = memref.alloc() : memref<200xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %4[%arg7] : memref<200xf64>
      memref.store %69, %52[%arg7] : memref<200xf64>
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %51[%arg7, %arg8] : memref<1000x200xf64>
        %70 = memref.load %52[%arg8] : memref<200xf64>
        %71 = arith.addf %69, %70 : f64
        memref.store %71, %52[%arg8] : memref<200xf64>
      }
    }
    %53 = memref.alloc() : memref<1000x200xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %3[%arg7, %arg8] : memref<1000x200xf64>
        memref.store %69, %53[%arg7, %arg8] : memref<1000x200xf64>
      }
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        %69 = memref.load %51[%arg7, %arg8] : memref<1000x200xf64>
        %70 = memref.load %53[%arg7, %arg8] : memref<1000x200xf64>
        %71 = arith.negf %69 : f64
        %72 = arith.mulf %71, %cst_0 : f64
        %73 = arith.addf %72, %70 : f64
        memref.store %73, %53[%arg7, %arg8] : memref<1000x200xf64>
      }
    }
    memref.dealloc %51 : memref<1000x200xf64>
    memref.dealloc %15 : memref<200xf64>
    memref.dealloc %14 : memref<1000x200xf64>
    %54 = memref.alloc() : memref<200xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %37[%arg7] : memref<200xf64>
      %70 = memref.load %52[%arg7] : memref<200xf64>
      %71 = arith.addf %69, %70 : f64
      memref.store %71, %54[%arg7] : memref<200xf64>
    }
    memref.dealloc %37 : memref<200xf64>
    %55 = memref.alloc() : memref<200xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      %69 = memref.load %41[%arg7] : memref<200xf64>
      %70 = memref.load %52[%arg7] : memref<200xf64>
      %71 = arith.addf %69, %70 : f64
      memref.store %71, %55[%arg7] : memref<200xf64>
    }
    memref.dealloc %52 : memref<200xf64>
    memref.dealloc %41 : memref<200xf64>
    %56 = memref.alloc() : memref<1000x200x128xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %1[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          memref.store %69, %56[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
        }
      }
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %53[%arg7, %arg8] : memref<1000x200xf64>
          %70 = memref.load %56[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %71 = arith.addf %69, %70 : f64
          memref.store %71, %56[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
        }
      }
    }
    memref.dealloc %53 : memref<1000x200xf64>
    memref.dealloc %13 : memref<1000x200x128xf64>
    %57 = memref.alloc() : memref<200x128xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 {
        %69 = memref.load %0[%arg7, %arg8] : memref<200x128xf64>
        memref.store %69, %57[%arg7, %arg8] : memref<200x128xf64>
      }
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %9[%arg8, %arg9] : memref<200x128xf64>
          %70 = memref.load %11[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %71 = memref.load %12[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %72 = memref.load %56[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %73 = memref.load %57[%arg8, %arg9] : memref<200x128xf64>
          %74 = arith.mulf %69, %70 : f64
          %75 = arith.addf %74, %71 : f64
          %76 = arith.mulf %72, %75 : f64
          %77 = arith.mulf %72, %75 : f64
          %78 = arith.addf %76, %77 : f64
          %79 = arith.mulf %78, %70 : f64
          %80 = arith.addf %79, %73 : f64
          memref.store %80, %57[%arg8, %arg9] : memref<200x128xf64>
        }
      }
    }
    %58 = memref.alloc() : memref<200x128xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 {
        %69 = memref.load %44[%arg7, %arg8] : memref<200x128xf64>
        %70 = memref.load %57[%arg7, %arg8] : memref<200x128xf64>
        %71 = arith.addf %69, %70 : f64
        memref.store %71, %58[%arg7, %arg8] : memref<200x128xf64>
      }
    }
    memref.dealloc %57 : memref<200x128xf64>
    memref.dealloc %44 : memref<200x128xf64>
    %59 = memref.alloc() : memref<1000x200x128xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %1[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          memref.store %69, %59[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
        }
      }
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %9[%arg8, %arg9] : memref<200x128xf64>
          %70 = memref.load %11[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %71 = memref.load %12[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %72 = memref.load %56[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %73 = memref.load %59[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %74 = arith.mulf %69, %70 : f64
          %75 = arith.addf %74, %71 : f64
          %76 = arith.mulf %72, %75 : f64
          %77 = arith.mulf %72, %75 : f64
          %78 = arith.addf %76, %77 : f64
          %79 = arith.mulf %78, %69 : f64
          %80 = arith.addf %79, %73 : f64
          memref.store %80, %59[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
        }
      }
    }
    %60 = memref.alloc() : memref<1000x200x128xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %1[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          memref.store %69, %60[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
        }
      }
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %9[%arg8, %arg9] : memref<200x128xf64>
          %70 = memref.load %11[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %71 = memref.load %12[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %72 = memref.load %56[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %73 = memref.load %60[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %74 = arith.mulf %69, %70 : f64
          %75 = arith.addf %74, %71 : f64
          %76 = arith.mulf %72, %75 : f64
          %77 = arith.mulf %72, %75 : f64
          %78 = arith.addf %76, %77 : f64
          %79 = arith.addf %78, %73 : f64
          memref.store %79, %60[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
        }
      }
    }
    memref.dealloc %56 : memref<1000x200x128xf64>
    memref.dealloc %12 : memref<1000x200x128xf64>
    %61 = memref.alloc() : memref<200x8128xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 iter_args(%idx = %c0) -> index {
        %id_next = scf.for %arg9 = %c0 to %arg8 step %c1 iter_args(%idy = %idx) -> index {
          %69 = memref.load %2[%arg7, %idy] : memref<200x8128xf64>
          memref.store %69, %61[%arg7, %idy] : memref<200x8128xf64>
          %idz = arith.addi %idy, %c1 : index
          scf.yield %idz : index
        }
        scf.yield %id_next : index
      }
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 iter_args(%idx = %c0) -> index {
          %id_next = scf.for %arg10 = %c0 to %arg9 step %c1 iter_args(%idy = %idx) -> index {
            %69 = memref.load %11[%arg7, %arg8, %arg10] : memref<1000x200x128xf64>
            %70 = memref.load %60[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
            %71 = memref.load %61[%arg8, %idy] : memref<200x8128xf64>
            %72 = arith.mulf %70, %69 : f64
            %73 = arith.addf %72, %71 : f64
            memref.store %73, %61[%arg8, %idy] : memref<200x8128xf64>
            %idz = arith.addi %idy, %c1 : index
            scf.yield %idz : index
          }
          scf.yield %id_next : index
        }
      }
    }
    %62 = memref.alloc() : memref<200x8128xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 iter_args(%idx = %c0) -> index {
        %id_next = scf.for %arg9 = %c0 to %arg8 step %c1 iter_args(%idy = %idx) -> index {
          %69 = memref.load %43[%arg7, %idy] : memref<200x8128xf64>
          %70 = memref.load %61[%arg7, %idy] : memref<200x8128xf64>
          %71 = arith.addf %69, %70 : f64
          memref.store %71, %62[%arg7, %idy] : memref<200x8128xf64>
          %idz = arith.addi %idy, %c1 : index
          scf.yield %idz : index
        }
        scf.yield %id_next : index
      }
    }
    memref.dealloc %61 : memref<200x8128xf64>
    memref.dealloc %43 : memref<200x8128xf64>
    %63 = memref.alloc() : memref<1000x200x128xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %1[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          memref.store %69, %63[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
        }
      }
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 iter_args(%idx = %c0) -> index {
          %id_next = scf.for %arg10 = %c0 to %arg9 step %c1 iter_args(%idy = %idx) -> index {
            %69 = memref.load %arg3[%arg8, %idy] : memref<200x8128xf64>
            %70 = memref.load %60[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
            %71 = memref.load %63[%arg7, %arg8, %arg10] : memref<1000x200x128xf64>
            %72 = arith.mulf %70, %69 : f64
            %73 = arith.addf %72, %71 : f64
            memref.store %73, %63[%arg7, %arg8, %arg10] : memref<1000x200x128xf64>
            %idz = arith.addi %idy, %c1 : index
            scf.yield %idz : index
          }
          scf.yield %id_next : index
        }
      }
    }
    memref.dealloc %60 : memref<1000x200x128xf64>
    memref.dealloc %11 : memref<1000x200x128xf64>
    %64 = memref.alloc() : memref<1000x200x128xf64>
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %59[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %70 = memref.load %63[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %71 = arith.addf %69, %70 : f64
          memref.store %71, %64[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
        }
      }
    }
    memref.dealloc %63 : memref<1000x200x128xf64>
    memref.dealloc %59 : memref<1000x200x128xf64>
    %65 = memref.alloc() : memref<200x128xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 {
        %69 = memref.load %0[%arg7, %arg8] : memref<200x128xf64>
        memref.store %69, %65[%arg7, %arg8] : memref<200x128xf64>
      }
    }
    scf.for %arg7 = %c0 to %c1000 step %c1 {
      scf.for %arg8 = %c0 to %c25 step %c1 {
        scf.for %arg9 = %c0 to %c10 step %c1 {
          %69 = memref.load %64[%arg7, %arg8, %arg9] : memref<1000x200x128xf64>
          %70 = memref.load %65[%arg8, %arg9] : memref<200x128xf64>
          %71 = arith.negf %69 : f64
          %72 = arith.addf %71, %70 : f64
          memref.store %72, %65[%arg8, %arg9] : memref<200x128xf64>
        }
      }
    }
    memref.dealloc %64 : memref<1000x200x128xf64>
    %66 = memref.alloc() : memref<200x128xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 {
        %69 = memref.load %0[%arg7, %arg8] : memref<200x128xf64>
        memref.store %69, %66[%arg7, %arg8] : memref<200x128xf64>
      }
    }
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 {
        %69 = memref.load %55[%arg7] : memref<200xf64>
        %70 = memref.load %66[%arg7, %arg8] : memref<200x128xf64>
        %71 = arith.addf %69, %70 : f64
        memref.store %71, %66[%arg7, %arg8] : memref<200x128xf64>
      }
    }
    memref.dealloc %55 : memref<200xf64>
    %67 = memref.alloc() : memref<200x128xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 {
        %69 = memref.load %58[%arg7, %arg8] : memref<200x128xf64>
        %70 = memref.load %9[%arg7, %arg8] : memref<200x128xf64>
        %71 = arith.mulf %69, %70 : f64
        memref.store %71, %67[%arg7, %arg8] : memref<200x128xf64>
      }
    }
    memref.dealloc %58 : memref<200x128xf64>
    memref.dealloc %9 : memref<200x128xf64>
    %68 = memref.alloc() : memref<200x128xf64>
    scf.for %arg7 = %c0 to %c25 step %c1 {
      scf.for %arg8 = %c0 to %c10 step %c1 {
        %69 = memref.load %66[%arg7, %arg8] : memref<200x128xf64>
        %70 = memref.load %67[%arg7, %arg8] : memref<200x128xf64>
        %71 = arith.addf %69, %70 : f64
        memref.store %71, %68[%arg7, %arg8] : memref<200x128xf64>
      }
    }
    memref.dealloc %67 : memref<200x128xf64>
    memref.dealloc %66 : memref<200x128xf64>
    return %54, %65, %68, %62 : memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x8128xf64>
  }
  func @lagrad_gmm_compressed(%arg0: memref<200xf64>, %arg1: memref<200x128xf64>, %arg2: memref<200x128xf64>, %arg3: memref<200x8128xf64>, %arg4: memref<1000x128xf64>, %arg5: f64, %arg6: i64) -> (memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x8128xf64>) {
    %0:4 = call @__grad_gmm_objective(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x8128xf64>, memref<1000x128xf64>, f64, i64) -> (memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x8128xf64>)
    return %0#0, %0#1, %0#2, %0#3 : memref<200xf64>, memref<200x128xf64>, memref<200x128xf64>, memref<200x8128xf64>
  }
}

