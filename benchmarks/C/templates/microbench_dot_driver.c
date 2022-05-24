#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUM_RUNS 6
#define N {{n}}
// #define N 1024
double *deadbeef = (double *)0xdeadbeef;

extern F64Descriptor1D lagrad_dot(/*arg0=*/double *, double *, int64_t, int64_t,
                                  int64_t,
                                  /*arg1=*/double *, double *, int64_t, int64_t,
                                  int64_t);
extern F64Descriptor1D enzyme_mlir_dot(/*arg0=*/double *, double *, int64_t,
                                       int64_t, int64_t,
                                       /*arg1=*/double *, double *, int64_t,
                                       int64_t, int64_t);
extern double *enzyme_c_dot(int64_t, double *, double *);

void check_dot_err(double *dx, double *y, const char *application) {
  double err = 0.0;
  for (size_t i = 0; i < N; i++) {
    err += fabs(dx[i] - y[i]);
  }

  if (err > 1e-6) {
    printf("(%s) error: %f\n", application, err);
  }
}

typedef unsigned long (*dotBodyFunc)(double *x, double *y);

unsigned long collect_lagrad_dot(double *x, double *y) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor1D res = lagrad_dot(deadbeef, x, 0, N, 1, deadbeef, y, 0, N, 1);
  gettimeofday(&stop, NULL);

  check_dot_err(res.aligned, y, "LAGrad Dot Product");
  free(res.aligned);
  return timediff(start, stop);
}

unsigned long collect_enzyme_mlir_dot(double *x, double *y) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor1D res =
      enzyme_mlir_dot(deadbeef, x, 0, N, 1, deadbeef, y, 0, N, 1);
  gettimeofday(&stop, NULL);

  check_dot_err(res.aligned, y, "Enzyme/MLIR Dot Product");
  free(res.aligned);
  return timediff(start, stop);
}

unsigned long collect_enzyme_c_dot(double *x, double *y) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double *res = enzyme_c_dot(N, x, y);
  gettimeofday(&stop, NULL);

  check_dot_err(res, y, "Enzyme/C Dot Product");
  free(res);
  return timediff(start, stop);
}

int main() {
  double *x = malloc(N * sizeof(double));
  double *y = malloc(N * sizeof(double));
  random_init_d_2d(x, N, 1);
  random_init_d_2d(y, N, 1);

  dotBodyFunc funcs[] = {collect_lagrad_dot, collect_enzyme_mlir_dot,
                         collect_enzyme_c_dot};
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);
  for (size_t app = 0; app < num_apps; app++) {
    unsigned long results_df[NUM_RUNS];
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = (*funcs[app])(x, y);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free(x);
  free(y);
  return 0;
}
