#include "standaloneabi.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

typedef struct {
  Descriptor2D first;
  Descriptor1D second;
} TupleResult;

extern Descriptor1D grad_matvec(/*M=*/float *, float *, int64_t, int64_t,
                                int64_t, int64_t, int64_t, /*x=*/float *,
                                float *, int64_t, int64_t, int64_t);
// extern TupleResult enzyme_matvec(/*M=*/float *, float *, int64_t, int64_t,
//                                  int64_t, int64_t, int64_t, /*dM=*/float *,
//                                  float *, int64_t, int64_t, int64_t, int64_t,
//                                  int64_t, /*x=*/float *, float *, int64_t,
//                                  int64_t, int64_t, /*dx=*/float *, float *,
//                                  int64_t, int64_t, int64_t);

void checkFirstArg(float *da, float *x, size_t m, size_t n) {
  float total_err = 0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      total_err += da[i * n + j] - x[j];
    }
  }
  if (fabs(total_err) > 1e-9) {
    printf("Err: (first arg) %f\n", total_err);
  }
}

void checkSecondArg(float *dx, float *M, size_t m, size_t n) {
  float total_err = 0;
  for (size_t i = 0; i < n; i++) {
    total_err += dx[i];
    for (size_t j = 0; j < m; j++) {
      total_err -= M[j * n + i];
    }
  }
  if (fabs(total_err) > 1e-9) {
    printf("Err (second arg): %f\n", total_err);
  }
}

int main() {
  float *deadbeef = (float *)0xdeadbeef;
  const size_t NUM_RUNS = 50;
  const size_t NUM_WARMUPS = 20;
  const size_t TOTAL = NUM_RUNS + NUM_WARMUPS;
  const size_t m = 512;
  const size_t n = 1024;
  float *M = (float *)malloc(m * n * sizeof(float));
  // float M[m * n];
  // float x[n];
  float *x = (float *)malloc(n * sizeof(float));
  // random_init_2d(M, m, n);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      // M[i * n + j] = i * n + j;
      M[i * n + j] = 0.5;
    }
  }
  for (size_t i = 0; i < n; i++) {
    x[i] = i + 1;
  }

  unsigned long grad_results[TOTAL];
  // unsigned long enzyme_results[TOTAL];
  for (size_t i = 0; i < TOTAL; i++) {
    struct timeval stop, start;
    gettimeofday(&start, NULL);
    Descriptor1D res =
        grad_matvec(deadbeef, M, 0, m, n, 1, 1, deadbeef, x, 0, n, 1);
    gettimeofday(&stop, NULL);
    grad_results[i] = timediff(start, stop);

    // checkFirstArg(res.aligned, x, m, n);
    checkSecondArg(res.aligned, M, m, n);
    // print_farr(res.aligned, 10);
    free(res.aligned);
    // free(res.first.aligned);
    // free(res.second.aligned);
  }

  // for (size_t i = 0; i < TOTAL; i++) {
  //   struct timeval start, stop;
  //   float *dM = (float *)malloc(m * n * sizeof(float));
  //   float *dx = (float *)malloc(n * sizeof(float));
  //   for (size_t i = 0; i < m * n; i++) {
  //     dM[i] = 0.0;
  //   }
  //   for (size_t i = 0; i < n; i++) {
  //     dx[i] = 0.0;
  //   }

  //   gettimeofday(&start, NULL);
  //   TupleResult res =
  //       enzyme_matvec(deadbeef, M, 0, m, n, 1, 1, deadbeef, dM, 0, m, n, 1,
  //       1,
  //                     deadbeef, x, 0, n, 1, deadbeef, dx, 0, n, 1);
  //   gettimeofday(&stop, NULL);

  //   enzyme_results[i] = timediff(start, stop);

  //   checkFirstArg(res.first.aligned, x, m, n);
  //   // free(res.first.aligned);
  //   // free(res.second.aligned);
  //   free(dM);
  //   free(dx);
  // }

  float grad_mean = 0;
  // float enzyme_mean = 0;
  for (size_t i = NUM_WARMUPS; i < TOTAL; i++) {
    grad_mean += grad_results[i];
    // enzyme_mean += enzyme_results[i];
  }

  printf("Number of runs: %lu (%lu warmup runs)\n", NUM_RUNS, NUM_WARMUPS);
  printf("Mean grad result: %f\n", grad_mean / NUM_RUNS);
  free(M);
  free(x);
  // printf("Mean enzyme result: %f\n", enzyme_mean / NUM_RUNS);
  // printf("Relative speedup: %f\n", enzyme_mean / grad_mean);
}
