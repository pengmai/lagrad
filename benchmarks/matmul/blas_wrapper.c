#include "lagrad_utils.h"
#include <cblas.h>
#include <stdint.h>
#include <stdio.h>

// TODO: Should add this to LAGrad utils probably

void _mlir_ciface_dmatmul(F64Descriptor2D *A, F64Descriptor2D *B,
                          F64Descriptor2D *out) {
  size_t M = A->size_0, K = A->size_1, N = out->size_1;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
              A->aligned + A->offset, K, B->aligned + B->offset, N, 0.0,
              out->aligned + out->offset, N);
}

void _mlir_ciface_smatmul_grad_first(F64Descriptor2D *g, F64Descriptor2D *B,
                                     F64Descriptor2D *out) {
  size_t M = g->size_0, K = g->size_1, N = out->size_1;
  // for (size_t i = 0; i < M * N; i++) {
  //   out->aligned[i] = 0;
  // }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0,
              g->aligned + g->offset, K, B->aligned + B->offset, K, 0.0,
              out->aligned, N);
}
