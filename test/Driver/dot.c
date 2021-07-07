/**
 * A driver to call a kernel written in MLIR.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes;
  int64_t strides;
} Descriptor;

typedef struct {
  int64_t rank;
  Descriptor *descriptor;
} F32MemRef;

extern float mlir_sum(float *, float *, int64_t, int64_t, int64_t);
extern F32MemRef ddot(float *, float *, int64_t, int64_t, int64_t, float *,
                      float *, int64_t, int64_t, int64_t);
extern void denzyme_dot(int, float *, float *, float *, float *);

void random_init(float *arr, int size) {
  for (int i = 0; i < size; ++i) {
    arr[i] = (float)rand() / (float)RAND_MAX;
  }
}

static inline unsigned long timediff(struct timeval start, struct timeval stop) {
  return (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
}

void print_arr(unsigned long *arr, int n) {
  printf("[");
  for (int i = 0; i < n; i++) {
    printf("%lu", arr[i]);
    if (i != n - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

int main() {
  const int size = 32768;
  const int NUM_RUNS = 20;
  bool check_val = false;

  unsigned long grad_results[NUM_RUNS];
  unsigned long enzyme_results[NUM_RUNS];

  for (int run = 0; run < NUM_RUNS; run++) {
    float a[size];
    float b[size];
    random_init(a, size);
    random_init(b, size);

    float *deadbeef = (float *)0xdeadbeef;
    struct timeval stop, start;

    gettimeofday(&start, NULL);
    F32MemRef dot_grad = ddot(deadbeef, a, 0, size, 1, deadbeef, b, 0, size, 1);
    gettimeofday(&stop, NULL);

    grad_results[run] = timediff(start, stop);
    if (check_val) {
      float error = 0;
      for (int i = 0; i < dot_grad.descriptor->sizes; ++i) {
        error += fabs(dot_grad.descriptor->aligned[i] - b[i]);
      }
      printf("Grad total absolute error: %f\n", error);
    }

    gettimeofday(&start, NULL);
    float da[size] = {0};
    float db[size] = {0};
    denzyme_dot(size, a, da, b, db);
    gettimeofday(&stop, NULL);

    enzyme_results[run] = timediff(start, stop);
    if (check_val) {
      float error = 0;
      for (int i = 0; i < size; i++) {
        error += fabs(da[i] - b[i]);
      }
      printf("Enzyme total absolute error: %f\n", error);
    }
  }

  print_arr(grad_results, NUM_RUNS);
  print_arr(enzyme_results, NUM_RUNS);
}
