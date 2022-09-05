#include "nn.h"
#include <cblas.h>
#include <stdint.h>

void _mlir_ciface_smatvec(F32Descriptor2D *A, F32Descriptor1D *x,
                          F32Descriptor1D *out) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, A->size_0, A->size_1, 1.0,
              A->aligned + A->offset, A->stride_0, x->aligned + x->offset,
              x->stride, 1.0, out->aligned + out->offset, out->stride);
}

void _mlir_ciface_svecmat(F32Descriptor1D *x, F32Descriptor2D *A,
                          F32Descriptor1D *out) {
  // printf("calling svecmat\n");
  cblas_sgemv(CblasRowMajor, CblasTrans, A->size_0, A->size_1, 1.0,
              A->aligned + A->offset, A->stride_0, x->aligned + x->offset,
              x->stride, 1.0, out->aligned + out->offset, out->stride);
}

void _mlir_ciface_souter(F32Descriptor1D *x, F32Descriptor1D *y,
                         F32Descriptor2D *out) {
  // for (size_t i = 0; i < out->size_0 * out->size_1; i++) {
  //   out->aligned[i] = 0;
  // }

  // cblas_sger(CblasRowMajor, out->size_0, out->size_1, 1.0,
  //            x->aligned + x->offset, x->stride, y->aligned + y->offset,
  //            y->stride, out->aligned + out->offset, out->size_0);

  float *outoff = out->aligned + out->offset;
  float *xoff = x->aligned + x->offset;
  float *yoff = y->aligned + y->offset;
  for (int i = 0; i < x->size; i++) {
    for (int j = 0; j < i; j++) {
      outoff[i * out->stride_0 + j] = xoff[i] * yoff[j];
    }
  }
}

void _mlir_ciface_smatmul(F32Descriptor2D *A, F32Descriptor2D *B,
                          F32Descriptor2D *out) {
  size_t M = A->size_0, K = A->size_1, N = out->size_1;
  // for (size_t i = 0; i < M * N; i++) {
  //   out->aligned[i] = 0;
  // }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
              A->aligned + A->offset, K, B->aligned + B->offset, N, 0.0,
              out->aligned + out->offset, N);
}

void _mlir_ciface_smatmul_grad_first(F32Descriptor2D *g, F32Descriptor2D *B,
                                     F32Descriptor2D *out) {
  size_t M = g->size_0, K = g->size_1, N = out->size_1;
  // for (size_t i = 0; i < M * N; i++) {
  //   out->aligned[i] = 0;
  // }

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0,
              g->aligned + g->offset, K, B->aligned + B->offset, K, 0.0,
              out->aligned, N);
}

void _mlir_ciface_smatmul_grad_second(F32Descriptor2D *A, F32Descriptor2D *g,
                                      F32Descriptor2D *out) {
  size_t M = A->size_1, K = A->size_0, N = out->size_1;
  // for (size_t i = 0; i < M * N; i++) {
  //   out->aligned[i] = 0;
  // }
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0,
              A->aligned + A->offset, M, g->aligned + g->offset, N, 0.0,
              out->aligned + out->offset, N);
}
