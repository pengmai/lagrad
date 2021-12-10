#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// #define N {{n}}
#define D {{d}}
#define NUM_RUNS 10
#define N 22
// #define D 128

double *deadbeef = (double *)0xdeadbeef;

extern void enzyme_cache_loop(
    /*A=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*dA=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*parents=*/int32_t *, int32_t *, int64_t, int64_t, int64_t,
    /*out=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*dout=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);

extern F64Descriptor2D manual_cache_loop(/*A=*/double *, double *, int64_t,
                                         int64_t, int64_t, int64_t, int64_t,
                                         /*parents=*/int32_t *, int32_t *,
                                         int64_t, int64_t, int64_t);

typedef unsigned long (*bodyFunc)(double *A, int32_t *parents, double *ref);

unsigned long enzyme_run_instance(double *A, int32_t *parents, double *ref) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double *dA = (double *)malloc(N * D * sizeof(double));
  double *out = (double *)malloc(N * D * sizeof(double));
  double *dout = (double *)malloc(N * D * sizeof(double));
  for (size_t i = 0; i < N * D; i++) {
    dA[i] = 0;
    out[i] = 0;
    dout[i] = 1;
  }
  enzyme_cache_loop(deadbeef, A, 0, N, D, D, 1, deadbeef, dA, 0, N, D, D, 1,
                    (int32_t *)deadbeef, parents, 0, N, 1, deadbeef, out, 0, N,
                    D, D, 1, deadbeef, dout, 0, N, D, D, 1);
  gettimeofday(&stop, NULL);

  for (size_t i = 0; i < N * D; i++) {
    ref[i] = dA[i];
  }

  free(dA);
  free(out);
  free(dout);

  return timediff(start, stop);
}

unsigned long manual_run_instance(double *A, int32_t *parents, double *ref) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor2D manual_res = manual_cache_loop(
      deadbeef, A, 0, N, D, D, 1, (int32_t *)deadbeef, parents, 0, N, 1);
  gettimeofday(&stop, NULL);

  double err = 0.0;
  for (size_t i = 0; i < N * D; i++) {
    err += fabs(manual_res.aligned[i] - ref[i]);
  }
  if (err > 1e-6) {
    printf("Manual err: %f\n", err);
  }

  free(manual_res.aligned);
  return timediff(start, stop);
}

int main() {
  double *A = (double *)malloc(N * D * sizeof(double));
  double *ref = (double *)malloc(N * D * sizeof(double));
  int parents[22] = {-1, 0,  1, 2,  3,  0,  5, 6,  7,  0,  9,
                     10, 11, 0, 13, 14, 15, 0, 17, 18, 19, 0};
  random_init_d_2d(A, N, D);

  bodyFunc funcs[] = {enzyme_run_instance, manual_run_instance};
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);

  // free(manual_res.aligned);
  unsigned long *results_df =
      (unsigned long *)malloc(NUM_RUNS * sizeof(unsigned long));

  for (size_t app = 0; app < num_apps; app++) {
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = (*funcs[app])(A, parents, ref);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }
  free(A);
  free(ref);
  free(results_df);
}
