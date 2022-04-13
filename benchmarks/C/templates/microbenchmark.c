#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#define NUM_RUNS 20
#define NUM_WARMUPS 4
#define N 1024
#define K 25
#define D 10

extern F64Descriptor2D mlir_elementwise_exp(double *, double *, int64_t,
                                            int64_t, int64_t, int64_t, int64_t);
extern F64Descriptor1D mlir_row_sum(double *, double *, int64_t, int64_t,
                                    int64_t, int64_t, int64_t);
extern F64Descriptor3D mlir_broadcasted_sub(
    /*arg0=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*arg1=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern F64Descriptor3D mlir_inner_einsum(
    /*arg0=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*arg1=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t);
extern F64Descriptor3D mlir_squared_mult(
    /*arg0=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*arg1=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*arg2=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t);
extern F64Descriptor2D mlir_batched_rowsum(double *, double *, int64_t, int64_t,
                                           int64_t, int64_t, int64_t, int64_t,
                                           int64_t);
extern F64Descriptor2D mlir_scalar_mult_sub(
    /*arg0=*/double *, double *, int64_t, int64_t, int64_t, /*arg1=*/
    double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern F64Descriptor1D mlir_matvec(
    /*arg0=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*arg1=*/double *, double *, int64_t, int64_t, int64_t,
    /*arg2=*/double *, double *, int64_t, int64_t, int64_t);
extern F64Descriptor1D mlir_tri_matvec(
    /*arg0=*/double *, double *, int64_t, int64_t, int64_t,
    /*arg1=*/double *, double *, int64_t, int64_t, int64_t,
    /*arg2=*/double *, double *, int64_t, int64_t, int64_t);
extern void c_elementwise_exp(double *, double *, size_t, size_t);
extern void c_rowsum(double *, double *, size_t, size_t);
extern void blas_rowsum(double *, double *, size_t, size_t);
extern void blas_inner_einsum(double *, double *, double *, size_t, size_t,
                              size_t);

void check_exp(double *before, double *after, int64_t m, int64_t n) {
  double err = 0.0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      err += fabs(after[i * n + j] - exp(before[i * n + j]));
    }
  }
  if (err > 1e-6) {
    printf("Elementwise Exp err: %f\n", err);
  }
}

void make_ltri(double *before, int64_t m, int64_t n) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      if (i <= j) {
        before[i * n + j] = 0;
      }
    }
  }
}

void compress_ltri(double *before, double *out, int64_t n) {
  size_t idx = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      if (i > j) {
        out[idx++] = before[i * n + j];
      }
    }
  }
}

void check_matvec(double *M, double *v, double *out, int64_t m, int64_t n) {
  double error = 0.0;
  for (size_t i = 0; i < m; i++) {
    double entry = 0.0;
    for (size_t j = 0; j < n; j++) {
      entry += M[i * n + j] * v[j];
    }
    error += fabs(entry - out[i]);
  }
  if (error > 1e-6) {
    printf("matvec error: %f\n", error);
  }
}

void check_rowsum(double *before, double *after, int64_t m, int64_t n) {
  double err = 0.0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      err += before[i * n + j];
    }
    err -= after[i];
  }
  if (fabs(err) > 1e-6) {
    printf("rowsum err: %f\n", err);
  }
}

/*
#map5 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
*/
void check_inner_einsum(double *arg0, double *arg1, double *after, int64_t n,
                        int64_t k, int64_t d) {
  double err = 0.0;
  double *naive_res = (double *)malloc(n * k * d * sizeof(double));
  for (size_t i = 0; i < n * k * d; i++) {
    naive_res[i] = 0.0;
  }

  for (size_t d0 = 0; d0 < n; d0++) {
    for (size_t d1 = 0; d1 < k; d1++) {
      for (size_t d2 = 0; d2 < d; d2++) {
        for (size_t d3 = 0; d3 < d; d3++) {
          naive_res[d0 * k * d + d1 * d + d2] +=
              arg0[d1 * d * d + d2 * d + d3] * arg1[d0 * k * d + d1 * d + d3];
        }
      }
    }
  }

  for (size_t i = 0; i < n * k * d; i++) {
    err += fabs(naive_res[i] - after[i]);
  }

  if (err > 1e-6) {
    printf("inner einsum err: %f\n", err);
  }
  free(naive_res);
}

int main() {
  unsigned long mlir_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long tri_results[NUM_WARMUPS + NUM_RUNS];
  // unsigned long c_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long blas_results[NUM_WARMUPS + NUM_RUNS];
  double *deadbeef = (double *)0xdeadbeef;
  // double *B = malloc(N * D * sizeof(double));
  // double *A = malloc(K * D * sizeof(double));
  // double *einsum_arg0 = malloc(K * D * D * sizeof(double));
  // double *einsum_arg1 = malloc(N * K * D * sizeof(double));
  srand(40000);
  double *mv1 = (double *)malloc(N * N * sizeof(double));
  double *mv2 = (double *)malloc(N * sizeof(double));
  double *out = (double *)malloc(N * sizeof(double));
  const int L_SIZE = N * (N - 1) / 2;
  double *ltri = (double *)malloc(L_SIZE * sizeof(double));
  random_init_d_2d(mv1, N, N);
  random_init_d_2d(mv2, N, 1);
  make_ltri(mv1, N, N);
  compress_ltri(mv1, ltri, N);

  // random_init_d_2d(einsum_arg0, K, D * D);
  // random_init_d_2d(einsum_arg1, N, K * D);
  // double *scalar_mult_arg0 = malloc(K * sizeof(double));
  // random_init_d_2d(scalar_mult_arg0, K, 1);
  // double *scalar_mult_arg1 = malloc(N * K * sizeof(double));
  // random_init_d_2d(scalar_mult_arg1, N, K);

  // double *squared_mult_arg0 = malloc(K * D * sizeof(double));
  // double *squared_mult_arg1 = malloc(N * K * D * sizeof(double));
  // double *squared_mult_arg2 = malloc(N * K * D * sizeof(double));
  // random_init_d_2d(squared_mult_arg0, K, D);
  // random_init_d_2d(squared_mult_arg1, N, K * D);
  // random_init_d_2d(squared_mult_arg2, N, K * D);
  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // F64Descriptor2D ans = mlir_elementwise_exp(deadbeef, A, 0, K, D, D, 1);
    // F64Descriptor3D ans = mlir_broadcasted_sub(deadbeef, B, 0, N, D, D, 1,
    //                                            deadbeef, A, 0, K, D, D, 1);
    // F64Descriptor3D ans =
    //     mlir_inner_einsum(deadbeef, einsum_arg0, 0, K, D, D, D * D, D, 1,
    //                       deadbeef, einsum_arg1, 0, N, K, D, K * D, D, 1);
    for (size_t i = 0; i < N; i++) {
      out[i] = 0;
    }

    F64Descriptor1D ans = mlir_matvec(deadbeef, mv1, 0, N, N, N, 1, deadbeef,
                                      mv2, 0, N, 1, deadbeef, out, 0, N, 1);
    // mlir_squared_mult(deadbeef, squared_mult_arg0, 0, K, D, D, 1, deadbeef,
    //                   squared_mult_arg1, 0, N, K, D, K * D, D, 1, deadbeef,
    //                   squared_mult_arg2, 0, N, K, D, K * D, D, 1);
    // mlir_batched_rowsum(deadbeef, squared_mult_arg1, 0, N, K, D, K * D, D,
    // 1);
    // mlir_scalar_mult_sub(deadbeef, scalar_mult_arg0, 0, K, 1, deadbeef,
    //                      scalar_mult_arg1, 0, N, K, K, 1);
    gettimeofday(&stop, NULL);

    mlir_results[run] = timediff(start, stop);
    check_matvec(mv1, mv2, ans.aligned, N, N);
    // check_inner_einsum(einsum_arg0, einsum_arg1, ans.aligned, N, K, D);
    // check_exp(A, ans.aligned, K, D);
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    for (size_t i = 0; i < N; i++) {
      out[i] = 0;
    }
    F64Descriptor1D ans =
        mlir_tri_matvec(deadbeef, ltri, 0, L_SIZE, 1, deadbeef, mv2, 0, N, 1,
                        deadbeef, out, 0, N, 1);
    gettimeofday(&stop, NULL);

    tri_results[run] = timediff(start, stop);
    check_matvec(mv1, mv2, ans.aligned, N, N);
  }

  // for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
  //   struct timeval start, stop;
  //   gettimeofday(&start, NULL);
  //   double *out = (double *)malloc(K * sizeof(double));
  //   for (size_t i = 0; i < K; i++) {
  //     out[i] = 0;
  //   }

  //   c_elementwise_exp(A, out, K, D);
  //   gettimeofday(&stop, NULL);

  //   c_results[run] = timediff(start, stop);
  //   check_exp(A, out, K, D);
  //   free(out);
  // }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // double *out = (double *)malloc(N * K * D * sizeof(double));
    // blas_inner_einsum(einsum_arg0, einsum_arg1, out, N, K, D);
    gettimeofday(&stop, NULL);

    blas_results[run] = timediff(start, stop);
    // check_inner_einsum(einsum_arg0, einsum_arg1, out, N, K, D);
    // free(out);
  }

  double mlir_sum = 0.0;
  double tri_sum = 0.0;
  // double c_sum = 0.0;
  // double blas_sum = 0.0;
  for (size_t run = NUM_WARMUPS; run < NUM_WARMUPS + NUM_RUNS; run++) {
    mlir_sum += mlir_results[run];
    tri_sum += tri_results[run];
    // c_sum += c_results[run];
    // blas_sum += blas_results[run];
  }

  printf("MLIR result: %f\n", mlir_sum / NUM_RUNS);
  printf("MLIR tri results: %f\n", tri_sum / NUM_RUNS);
  printf("Speedup from compression: %f\n", mlir_sum / tri_sum);
  // printf("C result: %f\n", c_sum / NUM_RUNS);
  // printf("BLAS result: %f\n", blas_sum / NUM_RUNS);
  // printf("Slowdown: %f\n", mlir_sum / c_sum);
  // free(A);
  free(ltri);
  free(mv1);
  free(mv2);
  free(out);
}
