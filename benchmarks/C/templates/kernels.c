#include "mlir_c_abi.h"
#include "cblas.h"
#include <stdint.h>
#include <stdlib.h>

// {% if args == [0] %}
float *c_dot(float *a, float *b, int64_t size) {
  float *out = (float *)malloc(size * sizeof(float));
  for (size_t i = 0; i < size; i++) {
    out[i] = b[i];
  }
  return out;
}

float *openblas_dot(float *a, float *b, int64_t size) {
  float *out = (float *)malloc(size * sizeof(float));
  cblas_scopy(size, b, 1, out, 1);
  return out;
}
// {% endif %}

void _mlir_ciface_linalg_dot_view{{n}}xf32_view{{n}}xf32_viewf32(
    F32Descriptor1D *a, F32Descriptor1D *b, F32Descriptor0D *out) {
  // TODO: Don't really need this function right now
}

void _mlir_ciface_linalg_copy_viewf32_viewf32(F32Descriptor0D *in,
                                              F32Descriptor0D *out) {
  out->aligned[0] = in->aligned[0];
}

void _mlir_ciface_sdot_grad_first(F32Descriptor0D *g, F32Descriptor1D *b,
                                  F32Descriptor1D *out) {
  cblas_scopy(out->size, b->aligned, 1, out->aligned, 1);
}
// void _mlir_ciface_sdot_grad_second();
