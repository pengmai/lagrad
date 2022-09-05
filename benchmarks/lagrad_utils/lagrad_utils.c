#include "lagrad_utils.h"
#include <stdio.h>

unsigned long timediff(struct timeval start, struct timeval stop) {
  return (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
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

void print_f_arr(float *arr, size_t n) {
  printf("[");
  for (size_t i = 0; i < n; i++) {
    printf("%.4e", arr[i]);
    if (i != n - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

void print_d_arr(const double *arr, size_t n) {
  printf("[");
  for (size_t i = 0; i < n; i++) {
    printf("%.4e", arr[i]);
    if (i != n - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

void print_f_arr_2d(float *arr, size_t m, size_t n) {
  printf("[\n");
  for (size_t i = 0; i < m; i++) {
    printf("  ");
    print_f_arr(arr + i * n, n);
  }
  printf("]\n");
}

void print_d_arr_2d(double *arr, size_t m, size_t n) {
  printf("[\n");
  for (size_t i = 0; i < m; i++) {
    printf("  ");
    print_d_arr(arr + i * n, n);
  }
  printf("]\n");
}

void print_d_arr_3d(double *arr, size_t m, size_t n, size_t k) {
  printf("[\n");
  for (size_t i = 0; i < m; i++) {
    printf("  [\n");

    for (size_t j = 0; j < n; j++) {
      printf("    ");
      print_d_arr(arr + (i * n * k) + (j * k), k);
    }

    printf("  ]\n");
  }
  printf("]\n");
}
