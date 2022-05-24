#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUM_RUNS 6
#define N {{n}}
double *deadbeef = (double *)0xdeadbeef;

extern F64Descriptor2D lagrad_matmul(/*A=*/double *, double *, int64_t, int64_t,
                                     int64_t, int64_t, int64_t,
                                     /*B=*/double *, double *, int64_t, int64_t,
                                     int64_t, int64_t, int64_t);
extern F64Descriptor2D enzyme_mlir_matmul(/*A=*/double *, double *, int64_t,
                                          int64_t, int64_t, int64_t, int64_t,
                                          /*B=*/double *, double *, int64_t,
                                          int64_t, int64_t, int64_t, int64_t);
extern double *enzyme_c_matmul(int64_t, double *, double *);

void check_matmul_grad(double *res, double *B, const char *application) {
  double err = 0.0;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      err += fabs(res[i * N + j] - B[j * N + i]);
    }
  }
  if (err > 1e-8) {
    printf("(%s) matmul error: %f\n", application, err);
  }
}

typedef unsigned long (*matmulBodyFunc)(double *, double *);

unsigned long collect_lagrad_matmul(double *A, double *B) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor2D res =
      lagrad_matmul(deadbeef, A, 0, N, N, N, 1, deadbeef, B, 0, N, N, N, 1);
  gettimeofday(&stop, NULL);
  check_matmul_grad(res.aligned, B, "LAGrad");
  free(res.aligned);
  return timediff(start, stop);
}

unsigned long collect_enzyme_mlir_matmul(double *A, double *B) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor2D res = enzyme_mlir_matmul(deadbeef, A, 0, N, N, N, 1, deadbeef,
                                           B, 0, N, N, N, 1);
  gettimeofday(&stop, NULL);
  check_matmul_grad(res.aligned, B, "Enzyme/MLIR");
  free(res.aligned);
  return timediff(start, stop);
}

unsigned long collect_enzyme_c_matmul(double *A, double *B) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double *res = enzyme_c_matmul(N, A, B);
  gettimeofday(&stop, NULL);
  check_matmul_grad(res, B, "Enzyme/C");
  free(res);
  return timediff(start, stop);
}

int main() {
  double *A = malloc(N * N * sizeof(double));
  double *B = malloc(N * N * sizeof(double));

  matmulBodyFunc funcs[] = {collect_lagrad_matmul, collect_enzyme_mlir_matmul,
                            collect_enzyme_c_matmul};
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);
  for (size_t app = 0; app < num_apps; app++) {
    unsigned long results_df[NUM_RUNS];
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = (*funcs[app])(A, B);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free(A);
  free(B);
}
