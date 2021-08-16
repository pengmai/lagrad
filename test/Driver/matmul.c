#include "standaloneabi.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

extern Descriptor2D grad_matmul(/*A=*/float *, float *, int64_t, int64_t,
                                int64_t, int64_t, int64_t, /*B=*/float *,
                                float *, int64_t, int64_t, int64_t, int64_t,
                                int64_t);

void check_matmul_first_arg(Descriptor2D C, float *B, size_t n, size_t k) {
  float broadcasted[n];
  for (size_t i = 0; i < n; i++) {
    broadcasted[i] = 0.0f;
    for (size_t j = 0; j < k; j++) {
      broadcasted[i] += B[i * k + j];
    }
  }
  float error = 0.0f;
  for (size_t i = 0; i < C.size_0; i++) {
    for (size_t j = 0; j < C.size_1; j++) {
      error += fabs(C.aligned[i * C.size_1 + j] - broadcasted[j]);
    }
  }
  if (error > 1e-9) {
    printf("Matmul first arg error: %f\n", error);
  }
}

int main() {
  const size_t m = 3;
  const size_t n = 4;
  const size_t k = 5;
  float *deadbeef = (float *)0xdeadbeef;
  float *A = (float *)malloc(m * n * sizeof(float));
  float *B = (float *)malloc(n * k * sizeof(float));

  for (size_t i = 0; i < m * n; i++) {
    A[i] = i;
  }
  for (size_t i = 0; i < n * k; i++) {
    B[i] = i;
  }

  Descriptor2D C =
      grad_matmul(deadbeef, A, 0, m, n, 1, 1, deadbeef, B, 0, n, k, 1, 1);

  printf("size: %llu x %llu\n", C.size_0, C.size_1);
  check_matmul_first_arg(C, B, n, k);
  print_2d(C.aligned, C.size_0, C.size_1);
}
