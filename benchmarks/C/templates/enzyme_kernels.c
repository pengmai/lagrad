#include "shared_types.h"
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

// {% if args == [0] %}
float *enzyme_dot_first(float *a, float *b, int64_t size) {
  float *da = (float *)malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) {
    da[i] = 0.0f;
  }
  __enzyme_autodiff(dot_primal, a, da, enzyme_const, b, size);
  return da;
}

// {% elif args == [1] %}
float *enzyme_dot_second(float *a, float *b, int64_t size) {
  float *db = (float *)malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) {
    db[i] = 0.0f;
  }
  __enzyme_autodiff(dot_primal, enzyme_const, a, b, db, size);
  return db;
}
// {% elif args == [0, 1] %}
RawDotGradient enzyme_dot_both(float *a, float *b, int64_t size) {
  float *da = (float *)malloc(size * sizeof(float));
  float *db = (float *)malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) {
    da[i] = 0.0f;
    db[i] = 0.0f;
  }
  __enzyme_autodiff(dot_primal, a, da, b, db, size);
  RawDotGradient result = {da, db};
  return result;
}
// {% endif %}
// {% endif %}
