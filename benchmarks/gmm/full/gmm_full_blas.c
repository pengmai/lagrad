/* Work in progress. Trying to get PyTorch's speed. */
#include "gmm_types.h"
#include "lagrad_utils.h"
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void blas_gmm_primal(int d, int k, int n, double *alphas, double *means,
                     double *Qs, double *Ls, double *x, double wishart_gamma,
                     int wishart_m, double *err) {
  // Elementwise exp
  double *Qdiags = malloc(d * k * sizeof(double));
  for (size_t i = 0; i < d * k; i++) {
    Qdiags[i] = exp(Qs[i]);
  }

  double *sum_qs = malloc(k * sizeof(double));
  print_d_arr_2d(Qs, k, d);
  double one[] = {1.0, 1.0};
  // double one = 1.0;
  // sum
  cblas_dgemv(CblasRowMajor, CblasNoTrans, k, d, 1.0, Qs, d, one, 0, 0.0,
              sum_qs, 1);
  print_d_arr(sum_qs, k);
  free(Qdiags);
  free(sum_qs);
}

void _mlir_ciface_dmatvec(F64Descriptor2D *A, F64Descriptor1D *x,
                          F64Descriptor1D *out) {
  cblas_dgemv(CblasRowMajor, CblasNoTrans, A->size_0, A->size_1, 1.0,
              A->aligned + A->offset, A->stride_0, x->aligned + x->offset,
              x->stride, 1.0, out->aligned + out->offset, out->stride);
}

void _mlir_ciface_svecmat(F64Descriptor1D *x, F64Descriptor2D *A,
                          F64Descriptor1D *out) {
  // printf("calling svecmat\n");
  // cblas_dgemv(CblasRowMajor, CblasTrans, A->size_0, A->size_1, 1.0,
  //             A->aligned + A->offset, A->stride_0, x->aligned + x->offset,
  //             x->stride, 1.0, out->aligned + out->offset, out->stride);
}

void _mlir_ciface_souter(F64Descriptor1D *x, F64Descriptor1D *y,
                         F64Descriptor2D *out) {

  double *outoff = out->aligned + out->offset;
  double *xoff = x->aligned + x->offset;
  double *yoff = y->aligned + y->offset;
  // for (int i = 0; i < x->size; i++) {
  //   for (int j = 0; j < i; j++) {
  //     outoff[i * out->stride_0 + j] = xoff[i] * yoff[j];
  //   }
  // }
}
