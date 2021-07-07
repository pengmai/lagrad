/**
 * A driver to call a kernel written in MLIR.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef long long int64_t;

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

void random_init(float *arr, int size) {
  for (int i = 0; i < size; ++i) {
    arr[i] = (float)rand() / (float)RAND_MAX;
  }
}

int main() {
  const int size = 32768;
  float a[size];
  float b[size];
  random_init(a, size);
  random_init(b, size);

  float *deadbeef = (float *)0xdeadbeef;
  struct timeval stop, start;

  gettimeofday(&start, NULL);
  F32MemRef dot_grad = ddot(deadbeef, a, 0, size, 1, deadbeef, b, 0, size, 1);
  gettimeofday(&stop, NULL);

  printf("Grad took %lu us\n",
         (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
  float error = 0;
  for (int i = 0; i < dot_grad.descriptor->sizes; ++i) {
    error += fabs(dot_grad.descriptor->aligned[i] - b[i]);
  }
  printf("Total absolute error: %f\n", error);
}
