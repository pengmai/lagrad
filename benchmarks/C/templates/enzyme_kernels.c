#include "shared_types.h"
// #include <stdint.h>
// #include <stdlib.h>

extern void *malloc(unsigned long);
extern void free(void *);
extern float __enzyme_autodiff(void *, ...);
typedef long long int64_t;
int enzyme_const;
int enzyme_dupnoneed;

// {% if application == "trimatvec" %}
void enzyme_trimatvec_dense_primal(double *M, double *x, double *out,
                                   int64_t N) {
  for (int i = 0; i < N; i++) {
    out[i] = 0;
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      out[i] += M[i * N + j] * x[j];
    }
  }
}

void enzyme_trimatvec_tri_primal(double *M, double *x, double *out, int64_t N) {
  for (int i = 0; i < N; i++) {
    out[i] = 0;
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < i; j++) {
      out[i] += M[i * N + j] * x[j];
    }
  }
}

void enzyme_trimatvec_compressed_primal(double *icf, double *x, double *out,
                                        int64_t N) {
  for (int i = 0; i < N; i++) {
    out[i] = 0;
  }
  int icf_idx = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < i; j++) {
      out[i] += icf[icf_idx] * x[j];
      icf_idx++;
    }
  }
}

void enzyme_trimatvec_dense_adjoint(double *M, double *dM, double *x,
                                    double *dx, double *out, double *dout,
                                    int64_t N) {
  __enzyme_autodiff(enzyme_trimatvec_dense_primal, M, dM, x, dx,
                    enzyme_dupnoneed, out, dout, N);
}

void enzyme_trimatvec_tri_adjoint(double *M, double *dM, double *x, double *dx,
                                  double *out, double *dout, int64_t N) {
  __enzyme_autodiff(enzyme_trimatvec_tri_primal, M, dM, x, dx, enzyme_dupnoneed,
                    out, dout, N);
}

void enzyme_trimatvec_compressed_adjoint(double *icf, double *dicf, double *x,
                                  double *dx, double *out, double *dout,
                                  int64_t N) {
  __enzyme_autodiff(enzyme_trimatvec_compressed_primal, icf, dicf, x, dx,
                    enzyme_dupnoneed, out, dout, N);
}

// {% else %}

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

// {% if application == "matmul" %}
float matmul_primal(float *A, float *B, int64_t M, int64_t N, int64_t K) {
  float *C = (float *)malloc(M * K * sizeof(float));
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      C[i * K + j] = 0.0;
    }
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      for (int k = 0; k < N; k++) {
        C[i * K + j] += A[i * N + k] * B[k * K + j];
      }
    }
  }

  float sum = 0.0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      sum += C[i * K + j];
    }
  }

  free(C);
  return sum;
}
// {% if args == [0] %}
float *enzyme_matmul_first(float *A, float *B, int64_t M, int64_t N,
                           int64_t K) {
  float *dA = (float *)malloc(M * N * sizeof(float));
  int size_a = M * N;
  for (int i = 0; i < size_a; i++) {
    dA[i] = 0.0;
  }

  __enzyme_autodiff(matmul_primal, A, dA, enzyme_const, B, M, N, K);
  return dA;
}
// {% endif %}
// {% endif %}

float matvec_primal(float *A, float *x, int64_t M, int64_t N) {
  float *out = (float *)malloc(M * sizeof(float));
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      out[i] += A[i * N + j] * x[j];
    }
  }

  float sum = 0;
  for (int i = 0; i < M; i++) {
    sum += out[i];
  }

  free(out);
  return sum;
}

float *enzyme_matvec_first(float *A, float *x, int64_t M, int64_t N) {
  float *dA = (float *)malloc(M * N * sizeof(float));
  int size_a = M * N;
  for (int i = 0; i < size_a; i++) {
    dA[i] = 0.0;
  }

  __enzyme_autodiff(matvec_primal, A, dA, enzyme_const, x, M, N);
  return dA;
}

float vecmat_primal(float *x, float *A, int64_t M, int64_t N) {
  float *out = (float *)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    out[i] = 0;
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      out[j] += A[i * N + j] * x[i];
    }
  }

  float sum = 0;
  for (int i = 0; i < N; i++) {
    sum += out[i];
  }

  free(out);
  return sum;
}

float *enzyme_vecmat_first(float *x, float *A, int64_t M, int64_t N) {
  float *dx = (float *)malloc(M * sizeof(float));
  for (int i = 0; i < M; i++) {
    dx[i] = 0;
  }

  __enzyme_autodiff(vecmat_primal, x, dx, enzyme_const, A, M, N);
  return dx;
}

// {% endif %} # application matvec