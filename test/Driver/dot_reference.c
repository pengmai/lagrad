/**
 * An attempt to understand the cost of allocating intermediate buffers for
 * every move in the unoptimized Grad dot program.
 */

#include "standaloneabi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

F32MemRef grad_dot_naive(int64_t size, float *a, float *b, float *zeroes) {
  float *deadbeef = (float *)0xdeadbeef;

  float *gradient_signal = (float *)malloc(size * sizeof(float));
  for (int64_t i = 0; i < size; i++) {
    gradient_signal[i] = 1.0f;
  }

  float *mul_result = (float *)malloc(size * sizeof(float));
  for (int64_t i = 0; i < size; i++) {
    mul_result[i] = b[i] * gradient_signal[i];
  }

  float *add_result = (float *)malloc(size * sizeof(float));
  for (int64_t i = 0; i < size; i++) {
    // In the LLVM source, this "0" is a constant marked zeroinitializer
    add_result[i] = mul_result[i] + zeroes[i];
  }

  float *dead_grad_signal = (float *)malloc(size * sizeof(float));
  for (int64_t i = 0; i < size; i++) {
    dead_grad_signal[i] = 1.0f;
  }

  float *dead_mul_result = (float *)malloc(size * sizeof(float));

  for (int64_t i = 0; i < size; i++) {
    dead_mul_result[i] = a[i] * dead_grad_signal[i];
  }

  Descriptor1D *da_desc = (Descriptor1D *)malloc(sizeof(Descriptor1D));
  da_desc->allocated = deadbeef;
  da_desc->aligned = add_result;
  da_desc->offset = 0;
  da_desc->size = size;
  da_desc->stride = 1;
  F32MemRef da = {.rank = 1, .descriptor = da_desc};
  return da;
}

F32MemRef grad_dot_stack(int64_t size, float *a, float *b, float *zeroes) {
  float *deadbeef = (float *)0xdeadbeef;

  float *gradient_signal = (float *)malloc(size * sizeof(float));
  for (int64_t i = 0; i < size; i++) {
    gradient_signal[i] = 1.0f;
  }

  // float *mul_result = (float *)malloc(size * sizeof(float));
  float mul_result[size];
  for (int64_t i = 0; i < size; i++) {
    mul_result[i] = b[i] * gradient_signal[i];
  }

  float add_result[size];
  for (int64_t i = 0; i < size; i++) {
    // In the LLVM source, this "0" is a constant marked zeroinitializer
    add_result[i] = mul_result[i] + zeroes[i];
  }

  float dead_grad_signal[size];
  for (int64_t i = 0; i < size; i++) {
    dead_grad_signal[i] = 1.0f;
  }

  float dead_mul_result[size];
  for (int64_t i = 0; i < size; i++) {
    dead_mul_result[i] = a[i] * dead_grad_signal[i];
  }

  Descriptor1D *da_desc = (Descriptor1D *)malloc(sizeof(Descriptor1D));
  da_desc->allocated = deadbeef;
  da_desc->aligned = add_result;
  da_desc->offset = 0;
  da_desc->size = size;
  da_desc->stride = 1;
  F32MemRef da = {.rank = 1, .descriptor = da_desc};
  return da;
}

static inline unsigned long timediff(struct timeval start,
                                     struct timeval stop) {
  return (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
}

int main() {
  int64_t size = 1024 * 8;
  float *zeroes = (float *)malloc(size * sizeof(float));
  memset(zeroes, 0, size * sizeof(float));

  float *a = (float *)malloc(size * sizeof(float));
  float *b = (float *)malloc(size * sizeof(float));
  random_init(a, size);
  random_init(b, size);
  const int NUM_RUNS = 51;
  unsigned long naive_results[NUM_RUNS];
  unsigned long noint_results[NUM_RUNS];

  for (int run = 0; run < NUM_RUNS; run++) {
    struct timeval stop, start;

    gettimeofday(&start, NULL);
    F32MemRef da = grad_dot_naive(size, a, b, zeroes);
    gettimeofday(&stop, NULL);
    naive_results[run] = timediff(start, stop);

    gettimeofday(&start, NULL);
    F32MemRef da_noint = grad_dot_stack(size, a, b, zeroes);
    gettimeofday(&stop, NULL);
    noint_results[run] = timediff(start, stop);
  }

  // printf("Naive results: ");
  // print_arr(naive_results, NUM_RUNS);
  // printf("Noint results: ");
  // print_arr(noint_results, NUM_RUNS);

  float result_sum = 0;
  for (int run = 1; run < NUM_RUNS; run++) {
    result_sum += naive_results[run];
  }
  printf("Mean naive result: %f\n", result_sum / (NUM_RUNS - 1));

  result_sum = 0;
  for (int run = 1; run < NUM_RUNS; run++) {
    result_sum += noint_results[run];
  }
  printf("Mean stack alloc result: %f\n", result_sum / (NUM_RUNS - 1));
  free(a);
  free(b);
  free(zeroes);
}
