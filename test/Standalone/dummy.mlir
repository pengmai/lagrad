// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() -> i32 {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = standalone.foo %{{.*}} : i32
        %res = standalone.foo %0 : i32
        %res2 = standalone.bar %res : i32
        return %res2 : i32
    }
}
