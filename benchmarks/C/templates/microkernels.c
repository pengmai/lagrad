#include <cblas.h>
#include <math.h>
#include <stdlib.h>

/**
 * out = exp(A)
 */
void c_elementwise_exp(double *A, double *out, size_t M, size_t N) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      out[i * N + j] = exp(A[i * N + j]);
    }
  }
}

void c_rowsum(double *A, double *out, size_t M, size_t N) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      out[i] += A[i * N + j];
    }
  }
}

void blas_rowsum(double *A, double *out, size_t M, size_t N) {
  double *x = malloc(N * sizeof(double));
  for (size_t i = 0; i < N; i++) {
    x[i] = 1.0;
  }

  cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0, A, N, x, 1, 0.0, out, 1);
  free(x);
}

/*
indexing_maps = [
  affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>,
  affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
  affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
]
iterator_types = ["parallel", "parallel", "parallel", "reduction"]

// What does matrix multiplication look like?
C[d0, d1] += A[d0, d2] * B[d2, d1]
*/
void blas_inner_einsum(double *arg0, double *arg1, double *out, size_t N,
                       size_t K, size_t D) {
  for (size_t i = 0; i < N * K * D; i++) {
    out[i] = 0.0;
  }

  for (size_t d0 = 0; d0 < N; d0++) {
    for (size_t d1 = 0; d1 < K; d1++) {
      for (size_t d2 = 0; d2 < D; d2++) {
        for (size_t d3 = 0; d3 < D; d3++) {
          // out[d0, d1, d2] += arg0[d1, d2, d3] * arg1[d0, d1, d3]
          out[d0 * K * D + d1 * D + d2] +=
              arg0[d1 * D * D + d2 * D + d3] * arg1[d0 * K * D + d1 * D + d3];
        }
      }
    }
  }
}
