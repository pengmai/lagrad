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