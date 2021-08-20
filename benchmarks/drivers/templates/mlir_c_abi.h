#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} F32Descriptor1D;

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size_0;
  int64_t size_1;
  int64_t stride_0;
  int64_t stride_1;
} F32Descriptor2D;

typedef struct {
  F32Descriptor1D da;
  F32Descriptor1D db;
} DotGradient;

typedef struct {
  F32Descriptor2D da;
  F32Descriptor2D db;
} MatVecGradient;

unsigned long timediff(struct timeval start, struct timeval stop) {
  return (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
}

void random_init(float *arr, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    arr[i] = (float)rand() / (float)RAND_MAX;
  }
}

void random_init_2d(float *arr, size_t m, size_t n) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      arr[i * n + j] = (float)rand() / (float)RAND_MAX;
    }
  }
}

void print_ul_arr(unsigned long *arr, size_t n) {
  printf("[");
  for (size_t i = 0; i < n; i++) {
    printf("%lu", arr[i]);
    if (i != n - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}
