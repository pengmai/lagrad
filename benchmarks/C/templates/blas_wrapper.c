#include "mlir_c_abi.h"
#include "shared_types.h"
#include <cblas.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void _mlir_ciface_dmatvec(F64Descriptor2D *A, F64Descriptor1D *x,
                          F64Descriptor1D *out) {
  cblas_dgemv(CblasRowMajor, CblasNoTrans, A->size_0, A->size_1, 1.0,
              A->aligned + A->offset, A->stride_0, x->aligned + x->offset,
              x->stride, 1.0, out->aligned + out->offset, out->stride);
}

void _mlir_ciface_ddot(F64Descriptor1D *x, F64Descriptor1D *y,
                       F64Descriptor0D *out) {
  out->aligned[out->offset] =
      cblas_ddot(x->size, x->aligned + x->offset, x->stride,
                 y->aligned + y->offset, y->stride);
  // Bear this special case in mind when x and y are the same memref
  // double res = cblas_dnrm2(x->size, x->aligned + x->offset, x->stride);
  // out->aligned[out->offset] = res * res;
}

void _mlir_ciface_sdot_grad_first(F32Descriptor0D *g, F32Descriptor1D *b,
                                  F32Descriptor1D *out) {
  // cblas_scopy(out->size, b->aligned, 1, out->aligned, 1);
  // cblas_sscal(out->size, g->aligned[0], out->aligned, 1);
}

void _mlir_ciface_sdot_grad_second(F32Descriptor0D *g, F32Descriptor1D *a,
                                   F32Descriptor1D *out) {
  // cblas_scopy(out->size, a->aligned, 1, out->aligned, 1);
  // cblas_sscal(out->size, g->aligned[0], out->aligned, 1);
}
