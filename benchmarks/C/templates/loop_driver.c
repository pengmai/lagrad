#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// #define N {{n}}
// #define K {{k}}
// #define D {{d}}
#define N 1000
#define K 25
#define D 10

double *deadbeef = (double *)0xdeadbeef;
extern double lg_nested_loop(/*A=*/double *, double *, int64_t, int64_t,
                             int64_t, int64_t, int64_t,
                             /*B=*/double *, double *, int64_t, int64_t,
                             int64_t, int64_t, int64_t);
extern double lg_loop(/*A=*/double *, double *, int64_t, int64_t, int64_t,
                      int64_t, int64_t);
extern F64Descriptor2D lagrad_loop(/*A=*/double *, double *, int64_t, int64_t,
                                   int64_t, int64_t, int64_t);
extern F64Descriptor2D lagrad_nested_loop(/*A=*/double *, double *, int64_t,
                                          int64_t, int64_t, int64_t, int64_t,
                                          /*B=*/double *, double *, int64_t,
                                          int64_t, int64_t, int64_t, int64_t);
extern F64Descriptor2D lagrad_main_term(/*A=*/double *, double *, int64_t,
                                        int64_t, int64_t, int64_t, int64_t,
                                        /*B=*/double *, double *, int64_t,
                                        int64_t, int64_t, int64_t, int64_t);
extern double en_nested_loop(/*A=*/double *, double *, int64_t, int64_t,
                             int64_t, int64_t, int64_t,
                             /*B=*/double *, double *, int64_t, int64_t,
                             int64_t, int64_t, int64_t);
extern F64Descriptor2D enzyme_main_term(/*A=*/double *, double *, int64_t,
                                        int64_t, int64_t, int64_t, int64_t,
                                        /*B=*/double *, double *, int64_t,
                                        int64_t, int64_t, int64_t, int64_t);
extern F64Descriptor2D enzyme_nested_loop(/*A=*/double *, double *, int64_t,
                                          int64_t, int64_t, int64_t, int64_t,
                                          /*B=*/double *, double *, int64_t,
                                          int64_t, int64_t, int64_t, int64_t);
extern F64Descriptor2D enzyme_loop(/*A=*/double *, double *, int64_t, int64_t,
                                   int64_t, int64_t, int64_t);

unsigned long mlir_primal(double *A, double *B) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  lg_nested_loop(deadbeef, A, 0, N, D, D, 1, deadbeef, B, 0, K, D, D, 1);
  gettimeofday(&stop, NULL);
  return timediff(start, stop);
}

unsigned long lagrad_adjoint(double *A, double *B) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor2D res = lagrad_loop(deadbeef, A, 0, N, D, D, 1);
  gettimeofday(&stop, NULL);
  double err = 0.0;
  for (size_t i = 0; i < res.size_0 * res.size_1; i++) {
    err += fabs(res.aligned[i] - 1.0);
  }

  if (err > 1e-6) {
    printf("LAGrad error: %f\n", err);
  }
  free(res.aligned);
  return timediff(start, stop);
}

unsigned long lagrad_main_term_adjoint(double *A, double *dA, double *B) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor2D res =
      lagrad_main_term(deadbeef, A, 0, N, D, D, 1, deadbeef, B, 0, K, D, D, 1);
  gettimeofday(&stop, NULL);

  double err = 0.0;
  for (size_t i = 0; i < res.size_0 * res.size_1; i++) {
    err += fabs(res.aligned[i] - dA[i]);
  }
  if (err > 1e-6) {
    printf("(LAGrad) Main Term Error: %f\n", err);
  }

  free(res.aligned);
  return timediff(start, stop);
}

unsigned long enzyme_main_term_adjoint(double *A, double *dA, double *B) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor2D res =
      enzyme_main_term(deadbeef, A, 0, N, D, D, 1, deadbeef, B, 0, K, D, D, 1);
  gettimeofday(&stop, NULL);
  for (size_t i = 0; i < res.size_0 * res.size_1; i++) {
    dA[i] = res.aligned[i];
  }

  free(res.aligned);
  return timediff(start, stop);
}

unsigned long lagrad_nested_adjoint(double *A, double *dA, double *B) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor2D res = lagrad_nested_loop(deadbeef, A, 0, N, D, D, 1, deadbeef,
                                           B, 0, K, D, D, 1);
  gettimeofday(&stop, NULL);
  double err = 0.0;
  for (size_t i = 0; i < res.size_0 * res.size_1; i++) {
    err += fabs(dA[i] - res.aligned[i]);
  }
  if (err > 1e-6) {
    printf("Nested loops LAGrad error: %f\n", err);
  }
  free(res.aligned);
  return timediff(start, stop);
}

unsigned long enzyme_primal(double *A, double *B) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  en_nested_loop(deadbeef, A, 0, N, D, D, 1, deadbeef, B, 0, K, D, D, 1);
  gettimeofday(&stop, NULL);
  return timediff(start, stop);
}

unsigned long enzyme_adjoint(double *A, double *B) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor2D res = enzyme_loop(deadbeef, A, 0, N, D, D, 1);
  gettimeofday(&stop, NULL);
  for (size_t i = 0; i < res.size_0 * res.size_1; i++) {
    if (fabs(res.aligned[i] - 1.0) > 1e-6) {
      printf("Element at idx %lu was not 1: %f\n", i, res.aligned[i]);
    }
  }

  free(res.aligned);
  return timediff(start, stop);
}

unsigned long enzyme_nested_adjoint(double *A, double *dA, double *B) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor2D res = enzyme_nested_loop(deadbeef, A, 0, N, D, D, 1, deadbeef,
                                           B, 0, K, D, D, 1);
  gettimeofday(&stop, NULL);
  double val = 0.0;
  for (size_t i = 0; i < res.size_0 * res.size_1; i++) {
    val += fabs(res.aligned[i]);
    dA[i] = res.aligned[i];
  }
  if (val < 1e-6) {
    printf("***Enzyme nested adjoint result was zero***\n");
  }

  free(res.aligned);
  return timediff(start, stop);
}

int main() {
  double *A = (double *)malloc(N * D * sizeof(double));
  double *dA = (double *)malloc(N * D * sizeof(double));
  double *B = (double *)malloc(K * D * sizeof(double));
  random_init_d_2d(A, N, D);
  random_init_d_2d(B, K, D);

  unsigned long time = mlir_primal(A, B);
  unsigned long etime = enzyme_nested_adjoint(A, dA, B);
  unsigned long ltime = lagrad_nested_adjoint(A, dA, B);
  // unsigned long etime = enzyme_main_term_adjoint(A, dA, B);
  // unsigned long ltime = lagrad_main_term_adjoint(A, dA, B);
  printf("Took: %lu vs %lu vs %lu\n", time, ltime, etime);

  free(A);
  free(dA);
  free(B);
}
