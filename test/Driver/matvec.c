#include "standaloneabi.h"
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

typedef struct {
  Descriptor2D first;
  Descriptor1D second;
} TupleResult;

extern TupleResult grad_matvec(/*M=*/float *, float *, int64_t, int64_t,
                               int64_t, int64_t, int64_t, /*x=*/float *,
                               float *, int64_t, int64_t, int64_t);

void checkFirstArg(float *da, float *x, size_t m, size_t n) {
  float total_err = 0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      total_err += da[i * n + j] - x[j];
    }
  }
  if (total_err > 1e-9) {
    printf("Err: %f\n", total_err);
  }
}

int main() {
  float *deadbeef = (float *)0xdeadbeef;
  const size_t NUM_RUNS = 50;
  const size_t NUM_WARMUPS = 20;
  const size_t TOTAL = NUM_RUNS + NUM_WARMUPS;
  const size_t m = 512;
  const size_t n = 1024;
  float M[m * n];
  float x[n];
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      M[i * n + j] = i * n + j;
    }
  }
  for (size_t i = 0; i < n; i++) {
    x[i] = i + 1;
  }

  unsigned long grad_results[TOTAL];
  for (size_t i = 0; i < TOTAL; i++) {
    struct timeval stop, start;
    gettimeofday(&start, NULL);
    TupleResult res =
        grad_matvec(deadbeef, M, 0, m, n, 1, 1, deadbeef, x, 0, n, 1);
    gettimeofday(&stop, NULL);
    grad_results[i] = timediff(start, stop);

    checkFirstArg(res.first.aligned, x, m, n);
    free(res.first.aligned);
    free(res.second.aligned);
  }

  float grad_mean = 0;
  for (size_t i = NUM_WARMUPS; i < TOTAL; i++) {
    grad_mean += grad_results[i];
  }

  printf("Number of runs: %lu (%lu warmup runs)\n", NUM_RUNS, NUM_WARMUPS);
  printf("Mean grad result: %f\n", grad_mean / NUM_RUNS);
  print_arr(grad_results, TOTAL);

  // // You have to use pointer arithmetic to access the sizes?
  // for (int i = 0; i < res.descriptor->sizes; i++) {
  //   for (int j = 0; j < res.descriptor->sizes + 1; j++) {
  //     printf("%f",
  //            res.descriptor->aligned[i * (res.descriptor->sizes + 1) + j]);
  //     if (j != n - 1) {
  //       printf(" ");
  //     }
  //   }
  //   printf("\n");
  // }
}
