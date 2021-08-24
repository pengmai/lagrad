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
RawDotGradient openblas_dot_both(float *a, float *b, int64_t size) {
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

float *c_matvec_first(float *A, float *x, float *g, int64_t M, int64_t N) {
  float *dA = (float *)malloc(M * N * sizeof(float));
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      dA[i * N + j] = g[i] * x[j];
    }
  }
  return dA;
}

float *openblas_matvec_first(float *A, float *x, float *g, int64_t M, int64_t N) {
  float *dA = (float *)malloc(M * N * sizeof(float));
  size_t size_a = M * N;
  for (size_t i = 0; i < size_a; i++) {
    dA[i] = 0.0;
  }

  cblas_sger(CblasRowMajor, M, N, 1.0, g, 1, x, 1, dA, M);
  return dA;
}
/**
 * Computes the operation GB^T
 * A: MxN
 * B: NxK
 * G: MxK
 */
float *c_matmul_first(float *A, float*B, float *G, int64_t M, int64_t N, int64_t K) {
  float *dA = (float *)malloc(M * N * sizeof(float));
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      dA[i * N + j] = 0.0;
    }
  }

  for (size_t i = 0; i < M; i++) {
    for (size_t k = 0; k < K; k++) {
      for (size_t j = 0; j < N; j++) {
        dA[i * N + j] += G[i * K + k] * B[j * K + k];
      }
    }
  }

  return dA;
}

float *openblas_matmul_first(float *A, float *B, float *G, int64_t M, int64_t N, int64_t K) {
  float *dA = (float *)malloc(M * N * sizeof(float));
  cblas_sscal(M * N, 0.0, dA, 1);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, G, M, B, K, 1.0, dA, N);
  return dA;
}

void _mlir_ciface_linalg_dot_view{{n}}xf32_view{{n}}xf32_viewf32(F32Descriptor1D *a, F32Descriptor1D *b, F32Descriptor0D *out) {
  out->aligned[0] = cblas_sdot(a->size, a->aligned, 1, b->aligned, 1);
}

void _mlir_ciface_linalg_matvec_view{{m}}x{{n}}xf32_view{{n}}xf32_view{{m}}xf32(F32Descriptor2D *A, F32Descriptor1D *x, F32Descriptor1D *out) {
  // Don't need this implementation
}

void _mlir_ciface_linalg_matmul_view{{m}}x{{n}}xf32_view{{n}}x{{k}}xf32_view{{m}}x{{k}}xf32(F32Descriptor2D *A, F32Descriptor2D *B, F32Descriptor2D *out) {
  // Don't need this implementation
}

void _mlir_ciface_linalg_copy_viewf32_viewf32(F32Descriptor0D *in,
                                              F32Descriptor0D *out) {
  out->aligned[0] = in->aligned[0];
}

void _mlir_ciface_linalg_copy_view{{m}}xf32_view{{m}}xf32(F32Descriptor1D *in, F32Descriptor1D *out) {
  cblas_scopy(out->size, in->aligned, 1, out->aligned, 1);
}

// For simplicity, this library only works with square matrices. Generating the
// right linalg.copy library calls without producing duplicates becomes trickier
// otherwise.
void _mlir_ciface_linalg_copy_view{{m}}x{{k}}xf32_view{{m}}x{{k}}xf32(F32Descriptor2D *in, F32Descriptor2D *out) {
  cblas_scopy(out->size_0 * out->size_1, in->aligned, 1, out->aligned, 1);
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

void _mlir_ciface_souter(F32Descriptor1D *x, F32Descriptor1D *y, F32Descriptor2D *out) {
  for (size_t i = 0; i < x->size; i++) {
    for (size_t j = 0; j < y->size; j++) {
      out->aligned[i * y->size + j] = x->aligned[i] * y->aligned[j];
    }
  }
}

void _mlir_ciface_smatmul_grad_first(F32Descriptor2D *g, F32Descriptor2D *B,
                                     F32Descriptor2D *out) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, out->size_0, out->size_1,
              g->size_1, 1.0, g->aligned, g->size_0, B->aligned, B->size_1, 1.0, out->aligned,
              out->size_0);
}

void _mlir_ciface_smatmul_grad_second(F32Descriptor2D *A, F32Descriptor2D *g,
                                      F32Descriptor2D *out) {}
