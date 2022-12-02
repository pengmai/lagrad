#include "cblas.h"
#include <stdint.h>
#include <stdio.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} Descriptor1D;

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size_0;
  int64_t size_1;
  int64_t stride_0;
  int64_t stride_1;
} Descriptor2D;

void _mlir_ciface_svecmat(Descriptor1D *v, Descriptor2D *m, Descriptor1D *out) {
  cblas_sgemv(CblasRowMajor, CblasTrans, m->size_0, m->size_1, 1.0, m->aligned,
              m->size_1, v->aligned, 1, 1.0, out->aligned, 1);
}

void _mlir_ciface_souter(Descriptor1D *x, Descriptor1D *y, Descriptor2D *out) {
  for (size_t i = 0; i < x->size; i++) {
    for (size_t j = 0; j < y->size; j++) {
      out->aligned[i * y->size + j] = x->aligned[i] * y->aligned[j];
    }
  }
  // for (size_t i = 0; i < out->size_0 * out->size_1; i++) {
  //   out->aligned[i] = 0;
  // }

  // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out->size_0,
  //             out->size_1, 1, 1.0, x->aligned, 1, y->aligned, out->size_1,
  //             0.0, out->aligned, out->size_1);
  // cblas_sger(CblasRowMajor, out->size_0, out->size_1, 1.0, x->aligned, 1,
  //            y->aligned, 1, out->aligned, out->size_1);
}

void _mlir_ciface_linalg_copy_view1024xf32_view1024xf32(Descriptor1D *in,
                                                        Descriptor1D *out) {
  cblas_scopy(in->size, in->aligned, in->stride, out->aligned, out->stride);
}

void _mlir_ciface_linalg_copy_view512xf32_view512xf32(Descriptor1D *in,
                                                      Descriptor1D *out) {
  cblas_scopy(in->size, in->aligned, in->stride, out->aligned, out->stride);
}

void _mlir_ciface_linalg_matvec_view512x1024xf32_view1024xf32_view512xf32(
    Descriptor2D *A, Descriptor1D *x, Descriptor2D *out) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, A->size_0, A->size_1, 1.0,
              A->aligned, A->size_1, x->aligned, 1, 1.0, out->aligned, 1);
}
