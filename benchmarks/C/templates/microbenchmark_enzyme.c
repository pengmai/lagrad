#define TARGET_OS_EMBEDDED 0
#include "shared_types.h"
#include <math.h>
#include <stdlib.h>

extern int enzyme_const;
extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern void __enzyme_autodiff(void *, ...);

// Returning a double here is slightly faster than taking a DPS output argument.
double edot(int64_t N, double *x, double *y) {
  double out = 0.0;
  for (int64_t i = 0; i < N; i++) {
    out += x[i] * y[i];
  }
  return out;
}

double *enzyme_c_dot(int64_t N, double *x, double *y) {
  double *dx = (double *)malloc(N * sizeof(double));
  for (int i = 0; i < N; i++) {
    dx[i] = 0;
  }
  double out = 0.0, dout = 1.0;

  __enzyme_autodiff(edot, N, enzyme_dup, x, dx, enzyme_const, y);
  return dx;
}

double earrmax(int64_t N, double *arr) {
  double max = arr[0];
  for (int64_t i = 1; i < N; i++) {
    if (arr[i] > max) {
      max = arr[i];
    }
  }
  return max;
}

double *enzyme_c_arrmax(int64_t N, double *arr) {
  double *darr = (double *)malloc(N * sizeof(double));
  for (int i = 0; i < N; i++) {
    darr[i] = 0;
  }
  __enzyme_autodiff(earrmax, N, arr, darr);
  return darr;
}

void evecadd(int64_t N, double *x, double *y, double *out) {
  for (int64_t i = 0; i < N; i++) {
    out[i] = x[i] + y[i];
  }
}

double *enzyme_c_vecadd(int64_t N, double *x, double *y, double *g) {
  double *dx = (double *)malloc(N * sizeof(double));
  double *out = (double *)malloc(N * sizeof(double));
  for (size_t i = 0; i < N; i++) {
    dx[i] = 0;
    out[i] = 0;
  }

  __enzyme_autodiff(evecadd, N, enzyme_dup, x, dx, enzyme_const, y,
                    enzyme_dupnoneed, out, g);
  free(out);
  return dx;
}

void evectorscalar(int64_t N, double *scal, double *arr, double *out) {
  for (int64_t i = 0; i < N; i++) {
    out[i] = arr[i] * *scal;
  }
}

double enzyme_c_vecscal(int64_t N, double scal, double *arr, double *dout) {
  double dscal;
  double *out = (double *)malloc(N * sizeof(double));
  for (int64_t i = 0; i < N; i++) {
    out[i] = 0;
  }

  __enzyme_autodiff(evectorscalar, N, enzyme_dup, &scal, &dscal, enzyme_const,
                    arr, enzyme_dupnoneed, out, dout);
  return dscal;
}

void ematmul(int64_t N, double *A, double *B, double *out) {
  for (int64_t i = 0; i < N; i++) {
    for (int64_t j = 0; j < N; j++) {
      for (int64_t k = 0; k < N; k++) {
        out[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}

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
