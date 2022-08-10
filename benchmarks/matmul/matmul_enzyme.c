#include <stdlib.h>

void ematmul(int64_t N, double *A, double *B, double *out) {
  for (int64_t i = 0; i < N; i++) {
    for (int64_t j = 0; j < N; j++) {
      for (int64_t k = 0; k < N; k++) {
        out[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}

extern int enzyme_const;
extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern void __enzyme_autodiff(void *, ...);
double *enzyme_c_matmul(int64_t N, double *A, double *B) {
  double *dA = (double *)malloc(N * N * sizeof(double));
  double *out = (double *)malloc(N * N * sizeof(double));
  double *dout = (double *)malloc(N * N * sizeof(double));
  for (int64_t i = 0; i < N * N; i++) {
    dA[i] = 0.0;
    out[i] = 0.0;
    dout[i] = 1.0;
  }

  __enzyme_autodiff(ematmul, N, enzyme_dup, A, dA, enzyme_const, B,
                    enzyme_dupnoneed, out, dout);
  free(out);
  free(dout);
  return dA;
}
