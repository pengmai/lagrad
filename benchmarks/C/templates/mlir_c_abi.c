#include "mlir_c_abi.h"
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

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

void uniform_init(float val, float *arr, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    arr[i] = val;
  }
}

void uniform_init_2d(float val, float *arr, size_t m, size_t n) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      arr[i * n + j] = val;
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
