#include <math.h>
#include <stdlib.h>

void ematmul(int64_t N, double *restrict A, double *restrict B, double *out) {
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

void iter_mul(int64_t N, int64_t M, double *x, double *out) {
  // double r = 0;
  double *intermediate = malloc(M * sizeof(double));
  // double sum;
  // for i in range(N):
  //   out += x * x
  for (int i = 0; i < N; i++) {
    // for (int j = 0; j < M; j++) {
    //   intermediate[j] = *x * *x;
    // }
    // sum = 0;
    for (int j = 0; j < M; j++) {
      intermediate[j] = x[j] * x[j];
      out[j] += intermediate[j];
      // sum += exp(intermediate[j]);
    }
    // r += log(sum);
  }
  // *out = r;
  free(intermediate);
}

// void partial_matmul(int64_t N, double *sc, double *B, double *out) {
//   double *A = calloc(N * N, sizeof(double));
//   A[0] = *sc;
//   ematmul(N, A, B, out);
//   free(A);
// }

double enzyme_example(int64_t N, int64_t M, double x) {
  double dx = 0.0, out = 0.0, dout = 1.0;
  __enzyme_autodiff(iter_mul, N, M, enzyme_dup, &x, &dx, &out, &dout);
  return dx;
}
