#include <stdint.h>
// #include <stdlib.h>

extern void *malloc(unsigned long);
extern float __enzyme_autodiff(void *, ...);
int enzyme_const;

// {% if application == "dot" %}
float dot_primal(float *a, float *b, int64_t size) {
  float res = 0.0f;
  for (int i = 0; i < size; i++) {
    res += a[i] * b[i];
  }
  return res;
}

float *enzyme_dot(float *a, float *b, int64_t size) {
  float *da = (float *)malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) {
    da[i] = 0.0f;
  }
  __enzyme_autodiff(dot_primal, a, da, enzyme_const, b, size);
  return da;
}
// {% endif %}
