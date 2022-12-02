#include "trmv.h"
#include <stdlib.h>

void ectrmv_packed(int64_t N, double *M, double *x, double *out) {
  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      int Lidx = j - (i + 1) + i * (2 * N - (i + 1)) / 2;
      out[j] += M[Lidx] * x[i];
    }
  }
}

extern int enzyme_const;
extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern void __enzyme_autodiff(void *, ...);
TRMVCompressedGrad enzyme_c_trmv_packed(int64_t N, double *M, double *x) {
  int tri_size = N * (N - 1) / 2;
  double *dM = calloc(tri_size, sizeof(double));
  double *dx = calloc(N, sizeof(double));
  double *out = malloc(N * sizeof(double));
  double *dout = malloc(N * sizeof(double));
  for (size_t i = 0; i < N; i++) {
    dout[i] = 1;
  }

  __enzyme_autodiff(ectrmv_packed, N, M, dM, x, dx, enzyme_dupnoneed, out,
                    dout);
  TRMVCompressedGrad grad = {.dM = {.allocated = NULL,
                                    .aligned = dM,
                                    .offset = 0,
                                    .size = tri_size,
                                    .stride = 1},
                             .dx = {.allocated = NULL,
                                    .aligned = dx,
                                    .offset = 0,
                                    .size = N,
                                    .stride = 1}};
  free(out);
  free(dout);
  return grad;
}
