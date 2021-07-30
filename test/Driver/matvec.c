#include "standaloneabi.h"
#include <stdint.h>
#include <stdio.h>
extern F32MemRef grad_matvec(/*M=*/float *, float *, int64_t, int64_t, int64_t,
                        int64_t, int64_t, /*x=*/float *, float *, int64_t,
                        int64_t, int64_t);

int main() {
  float *deadbeef = (float *)0xdeadbeef;
  const int m = 3;
  const int n = 4;
  float M[m * n];
  float x[n];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      M[i * n + j] = 1;
    }
  }
  for (int i = 0; i < n; i++) {
    x[i] = i + 1;
  }

  F32MemRef res = grad_matvec(deadbeef, M, 0, m, n, 1, 1, deadbeef, x, 0, n, 1);

  printf("rank: %llu\n", res.rank);
  // You have to use pointer arithmetic to access the sizes?
  for (int i = 0; i < res.descriptor->sizes; i++) {
    for (int j = 0; j < res.descriptor->sizes + 1; j++) {
      printf("%f",
             res.descriptor->aligned[i * (res.descriptor->sizes + 1) + j]);
      if (j != n - 1) {
        printf(" ");
      }
    }
    printf("\n");
  }
}
