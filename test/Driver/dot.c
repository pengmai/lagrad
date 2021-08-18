/**
 * A driver to call a kernel written in MLIR.
 */

#include "standaloneabi.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct {
  Descriptor1D da;
  Descriptor1D db;
} DotGradient;

extern DotGradient ddot(float *, float *, int64_t, int64_t, int64_t, float *,
                        float *, int64_t, int64_t, int64_t);
// extern void denzyme_dot(int, float *, float *, float *, float *);

// extern DotGradient enzyme_dot(float *, float *, int64_t, int64_t, int64_t,
//                               float *, float *, int64_t, int64_t, int64_t);

int main() {
  const int size = 131072;
  const int NUM_RUNS = 51;
  bool check_val = true;

  unsigned long grad_results[NUM_RUNS];
  unsigned long enzyme_results[NUM_RUNS];
  float *a = (float *)malloc(size * sizeof(float));
  float *b = (float *)malloc(size * sizeof(float));
  random_init(a, size);
  random_init(b, size);

  float *deadbeef = (float *)0xdeadbeef;
  for (int run = 0; run < NUM_RUNS; run++) {

    struct timeval start, stop;

    gettimeofday(&start, NULL);
    DotGradient dot_grad =
        ddot(deadbeef, a, 0, size, 1, deadbeef, b, 0, size, 1);
    gettimeofday(&stop, NULL);

    grad_results[run] = timediff(start, stop);
    if (check_val) {
      float error = 0;
      for (int i = 0; i < dot_grad.da.size; ++i) {
        error += fabs(dot_grad.da.aligned[i] - b[i]);
      }
      if (error > 1e-9) {
        printf("Grad total absolute error: %f\n", error);
      }
      error = 0;
      for (size_t i = 0; i < dot_grad.db.size; i++)
      {
        error += fabs(dot_grad.db.aligned[i] - a[i]);
      }
      if (error > 1e-9) {
        printf("Grad total absolute error (second arg): %f\n", error);
      }
    }
    free(dot_grad.da.aligned);
    free(dot_grad.db.aligned);
  }

  // for (int run = 0; run < NUM_RUNS; run++) {
  //   struct timeval start, stop;

  //   gettimeofday(&start, NULL);
  //   DotGradient edot_grad =
  //       enzyme_dot(deadbeef, a, 0, size, 1, deadbeef, b, 0, size, 1);
  //   gettimeofday(&stop, NULL);

  //   enzyme_results[run] = timediff(start, stop);
  //   if (check_val) {
  //     float error = 0;
  //     for (int i = 0; i < size; i++) {
  //       error += fabs(edot_grad.da.aligned[i] - b[i]);
  //     }
  //     if (error > 1e-9) {
  //       printf("Enzyme total absolute error: %f\n", error);
  //     }
  //     error = 0;
  //     for (size_t i = 0; i < size; i++) {
  //       error += fabs(edot_grad.db.aligned[i] - a[i]);
  //     }
  //     if (error > 1e-9) {
  //       printf("Enzyme total absolute error (second arg): %f\n", error);
  //     }
  //   }
  //   free(edot_grad.da.aligned);
  //   free(edot_grad.db.aligned);
  // }

  printf("Number of runs: %d (%d warmup run(s))\n", NUM_RUNS, 1);
  float grad_res = 0;
  for (int i = 1; i < NUM_RUNS; i++) {
    grad_res += grad_results[i];
  }
  printf("Mean grad result: %f\n", grad_res / (NUM_RUNS - 1));

  // float enzy_res = 0;
  // for (int i = 1; i < NUM_RUNS; i++) {
  //   enzy_res += enzyme_results[i];
  // }
  // printf("Mean enzyme result: %f\n", enzy_res / (NUM_RUNS - 1));
  // printf("Speedup: %f\n", enzy_res / grad_res);

  free(a);
  free(b);
}
