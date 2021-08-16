#include "standaloneabi.h"
#include <stdint.h>
#include <stdio.h>

extern F32MemRef enzyme_square(float *, float *, int64_t, int64_t, int64_t);

int main() {
  float a[1] = {-7.9};
  float *deadbeef = (float *)0xdeadbeef;
  F32MemRef res = enzyme_square(deadbeef, a, 0, 1, 1);
  printf("res: %f\n", res.descriptor->aligned[0]);
}
