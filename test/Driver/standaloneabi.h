#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} Descriptor1D;

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size_0;
  int64_t size_1;
  int64_t stride_0;
  int64_t stride_1;
} Descriptor2D;

typedef struct {
  int64_t rank;
  Descriptor1D *descriptor;
} F32MemRef;

void random_init(float *arr, int size) {
  for (int i = 0; i < size; ++i) {
    arr[i] = (float)rand() / (float)RAND_MAX;
  }
}

void random_init_2d(float *arr, int m, int n) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      arr[i * n + j] = (float)rand() / (float)RAND_MAX;
    }
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

void print_2d(float *arr, size_t m, size_t n) {
  printf("[");
  for (size_t i = 0; i < m; i++)
  {
    if (i != 0) {
      printf(" ");
    }
    printf("[");
    for (size_t j = 0; j < n; j++)
    {
      printf("%f", arr[i * n + j]);
      if (j != n - 1) {
        printf(", ");
      }
    }
    printf("]");
    if (i != m - 1) {
      printf("\n");
    }
  }
  printf("]\n");
}

void print_farr(float *arr, int n) {
  printf("[");
  for (int i = 0; i < n; i++) {
    printf("%f", arr[i]);
    if (i != n - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

unsigned long timediff(struct timeval start, struct timeval stop) {
  return (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
}
