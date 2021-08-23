#include "cblas.h"
#include "mlir_c_abi.h"
#include "shared_types.h"
#include <stdint.h>
#include <stdlib.h>

// {% if args == [0] %}
float *c_dot_first(float *a, float *b, float g, int64_t size) {
  float *out = (float *)malloc(size * sizeof(float));
  for (size_t i = 0; i < size; i++) {
    out[i] = b[i] * g;
  }
  return out;
}
// {% elif args == [1] %}
float *c_dot_second(float *a, float *b, float g, int64_t size) {
  float *out = (float *)malloc(size * sizeof(float));
  for (size_t i = 0; i < size; i++) {
    out[i] = a[i] * g;
  }
  return out;
}
// {% else %}
RawDotGradient c_dot_both(float *a, float *b, float g, int64_t size) {
  float *da = (float *)malloc(size * sizeof(float));
  float *db = (float *)malloc(size * sizeof(float));
  for (size_t i = 0; i < size; i++) {
    da[i] = b[i] * g;
    db[i] = a[i] * g;
  }

  RawDotGradient result = {da, db};
  return result;
}
// {% endif %}
// {% if args == [0] %}
float *openblas_dot_first(float *a, float *b, int64_t size) {
  float *out = (float *)malloc(size * sizeof(float));
  cblas_scopy(size, b, 1, out, 1);
  cblas_sscal(size, 1.0f, out, 1);
  return out;
}
// {% elif args == [1] %}
float *openblas_dot_second(float *a, float *b, int64_t size) {
  float *out = (float *)malloc(size * sizeof(float));
  cblas_scopy(size, a, 1, out, 1);
  cblas_sscal(size, 1.0f, out, 1);
  return out;
}
// {% else %}
RawDotGradient openblas_dot_both(float *a, float *b, int size) {
  float *da = (float *)malloc(size * sizeof(float));
  float *db = (float *)malloc(size * sizeof(float));
  cblas_scopy(size, b, 1, da, 1);
  cblas_sscal(size, 1.0f, da, 1);
  cblas_scopy(size, a, 1, db, 1);
  cblas_sscal(size, 1.0f, db, 1);
  RawDotGradient result = {da, db};
  return result;
}
// {% endif %}

void _mlir_ciface_linalg_dot_view{{n}}xf32_view{{n}}xf32_viewf32(F32Descriptor1D *a, F32Descriptor1D *b, F32Descriptor0D *out) {
  out->aligned[0] = cblas_sdot(a->size, a->aligned, 1, b->aligned, 1);
}

void _mlir_ciface_linalg_copy_viewf32_viewf32(F32Descriptor0D *in,
                                              F32Descriptor0D *out) {
  out->aligned[0] = in->aligned[0];
}

void _mlir_ciface_sdot_grad_first(F32Descriptor0D *g, F32Descriptor1D *b,
                                  F32Descriptor1D *out) {
  cblas_scopy(out->size, b->aligned, 1, out->aligned, 1);
  cblas_sscal(out->size, g->aligned[0], out->aligned, 1);
}

void _mlir_ciface_sdot_grad_second(F32Descriptor0D *g, F32Descriptor1D *a,
                                   F32Descriptor1D *out) {
  cblas_scopy(out->size, a->aligned, 1, out->aligned, 1);
  cblas_sscal(out->size, g->aligned[0], out->aligned, 1);
}
