#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

void random_init(float *arr, int size) {
  for (int i = 0; i < size; ++i) {
    arr[i] = (float)rand() / (float)RAND_MAX;
  }
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
