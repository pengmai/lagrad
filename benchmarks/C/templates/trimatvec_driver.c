#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUM_RUNS 3
#define NUM_WARMUPS 0
// #define N {{n}}
#define N 16
#define TRI_SIZE N *(N - 1) / 2

enum ResultCol {
  EnzymeDensePrimal,
  EnzymeDenseAdjoint,
  EnzymeTriPrimal,
  EnzymeTriAdjoint,
  EnzymeCompPrimal,
  EnzymeCompAdjoint,
  LAGradDensePrimal,
  LAGradDenseAdjoint,
  LAGradTriPrimal,
  LAGradTriAdjoint,
  LAGradCompPrimal,
  LAGradCompAdjoint,
};

typedef struct _f64matvecgrad {
  F64Descriptor2D dM;
  F64Descriptor1D dx;
} F64MatVecGradient;

extern void enzyme_trimatvec_dense_primal(double *M, double *x, double *out,
                                          int64_t n);
extern void enzyme_trimatvec_tri_primal(double *M, double *x, double *out,
                                        int64_t n);
extern void enzyme_trimatvec_compressed_primal(double *icf, double *x,
                                               double *out, int64_t n);
extern void enzyme_trimatvec_dense_adjoint(double *M, double *dM, double *x,
                                           double *dx, double *out,
                                           double *dout, int64_t n);
extern void enzyme_trimatvec_tri_adjoint(double *M, double *dM, double *x,
                                         double *dx, double *out, double *dout,
                                         int64_t n);
extern void enzyme_trimatvec_compressed_adjoint(double *icf, double *dicf,
                                                double *x, double *dx,
                                                double *out, double *dout,
                                                int64_t n);
extern F64Descriptor1D mlir_trimatvec_dense_primal(
    /*M=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, /*out=*/double *,
    double *, int64_t, int64_t, int64_t);
extern F64MatVecGradient mlir_trimatvec_dense_adjoint(
    /*M=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, /*out=*/double *,
    double *, int64_t, int64_t, int64_t);
extern F64Descriptor1D mlir_trimatvec_tri_primal(
    /*M=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, /*out=*/double *,
    double *, int64_t, int64_t, int64_t);
extern F64MatVecGradient mlir_trimatvec_tri_adjoint(
    /*M=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, /*out=*/double *,
    double *, int64_t, int64_t, int64_t);

void make_ltri(double *before, int64_t m, int64_t n) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      if (i <= j) {
        before[i * n + j] = 0;
      }
    }
  }
}

void convert_to_compressed(double *full, double *compressed) {
  size_t idx = 0;
  for (size_t i = 1; i < N; i++) {
    for (size_t j = 0; j < i; j++) {
      compressed[idx] = full[i * N + j];
      idx++;
    }
  }
}

void convert_to_full(double *compressed, double *full) {
  size_t idx = 0;
  for (size_t i = 1; i < N; i++) {
    for (size_t j = 0; j < i; j++) {
      full[i * N + j] = compressed[idx];
      idx++;
    }
  }
}

void check_matvec_primal(double *M, double *x, double *out,
                         const char *application) {
  double err = 0;
  for (size_t i = 0; i < N; i++) {
    double row = out[i];
    for (size_t j = 0; j < N; j++) {
      row -= (M[i * N + j] * x[j]);
    }
    err += fabs(row);
  }
  if (err > 1e-6) {
    printf("(%s) primal err: %f\n", application, err);
  }
}

void check_matvec_adjoint(double *M, double *dM, double *x, double *dx,
                          double *dout, const char *application) {
  double err = 0;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < i; j++) {
      err += fabs(dM[i * N + j] - dout[i] * x[j]);
    }
  }

  if (err > 1e-6) {
    printf("(%s) adjoint first arg err: %f\n", application, err);
  }

  err = 0;
  for (size_t i = 0; i < N; i++) {
    double row = dx[i];
    for (size_t j = 0; j < N; j++) {
      row -= M[j * N + i] * dout[j];
    }
    err += fabs(row);
  }

  if (err > 1e-6) {
    printf("(%s) adjoint second arg err: %f\n", application, err);
  }
}

double take_avg(unsigned long *arr, size_t n) {
  double avg = 0;
  for (size_t i = 0; i < n; i++) {
    avg += arr[i];
  }
  return avg / n;
}

typedef unsigned long (*bodyFunc)(double *M, double *dM, double *icf,
                                  double *dicf, double *x, double *dx,
                                  double *out, double *dout);

/* Implementations */
unsigned long enzyme_dense_primal(double *M, double *dM, double *icf,
                                  double *dicf, double *x, double *dx,
                                  double *out, double *dout) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  enzyme_trimatvec_dense_primal(M, x, out, N);
  gettimeofday(&stop, NULL);

  check_matvec_primal(M, x, out, "enzyme full dense");
  return timediff(start, stop);
}

unsigned long enzyme_dense_adjoint(double *M, double *dM, double *icf,
                                   double *dicf, double *x, double *dx,
                                   double *out, double *dout) {
  struct timeval start, stop;
  size_t M_size = N * N;
  gettimeofday(&start, NULL);
  for (size_t i = 0; i < M_size; i++) {
    dM[i] = 0;
  }
  for (size_t i = 0; i < N; i++) {
    out[i] = 0;
  }
  enzyme_trimatvec_dense_adjoint(M, dM, x, dx, out, dout, N);
  gettimeofday(&stop, NULL);

  check_matvec_adjoint(M, dM, x, dx, dout, "enzyme full dense");
  return timediff(start, stop);
}

unsigned long enzyme_tri_primal(double *M, double *dM, double *icf,
                                double *dicf, double *x, double *dx,
                                double *out, double *dout) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  enzyme_trimatvec_tri_primal(M, x, out, N);
  gettimeofday(&stop, NULL);

  check_matvec_primal(M, x, out, "enzyme full tri");
  return timediff(start, stop);
}

unsigned long enzyme_tri_adjoint(double *M, double *dM, double *icf,
                                 double *dicf, double *x, double *dx,
                                 double *out, double *dout) {
  struct timeval start, stop;
  size_t M_size = N * N;
  gettimeofday(&start, NULL);
  for (size_t i = 0; i < M_size; i++) {
    dM[i] = 0;
  }
  for (size_t i = 0; i < N; i++) {
    dx[i] = 0;
  }
  enzyme_trimatvec_tri_adjoint(M, dM, x, dx, out, dout, N);
  gettimeofday(&stop, NULL);
  check_matvec_adjoint(M, dM, x, dx, dout, "enzyme full tri");
  return timediff(start, stop);
}

unsigned long enzyme_compressed_primal(double *M, double *dM, double *icf,
                                       double *dicf, double *x, double *dx,
                                       double *out, double *dout) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  enzyme_trimatvec_compressed_primal(icf, x, out, N);
  gettimeofday(&stop, NULL);
  check_matvec_primal(M, x, out, "enzyme compressed");
  return timediff(start, stop);
}

unsigned long enzyme_compressed_adjoint(double *M, double *dM, double *icf,
                                        double *dicf, double *x, double *dx,
                                        double *out, double *dout) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  for (size_t i = 0; i < TRI_SIZE; i++) {
    dicf[i] = 0;
  }
  for (size_t i = 0; i < N; i++) {
    dx[i] = 0;
  }
  enzyme_trimatvec_compressed_adjoint(icf, dicf, x, dx, out, dout, N);
  gettimeofday(&stop, NULL);
  check_matvec_adjoint(M, dM, x, dx, dout, "enzyme compressed");
  return timediff(start, stop);
}

unsigned long mlir_dense_primal(double *M, double *dM, double *icf,
                                double *dicf, double *x, double *dx,
                                double *out, double *dout) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  for (size_t i = 0; i < N; i++) {
    out[i] = 0;
  }
  F64Descriptor1D ans = mlir_trimatvec_dense_primal(
      deadbeef, M, 0, N, N, N, 1, deadbeef, x, 0, N, 1, deadbeef, out, 0, N, 1);
  gettimeofday(&stop, NULL);
  check_matvec_primal(M, x, ans.aligned, "MLIR full dense");
  free(ans.aligned);
  return timediff(start, stop);
}

unsigned long mlir_dense_adjoint(double *M, double *dM, double *icf,
                                 double *dicf, double *x, double *dx,
                                 double *out, double *dout) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  for (size_t i = 0; i < N; i++) {
    out[i] = 0;
  }

  F64MatVecGradient ans = mlir_trimatvec_dense_adjoint(
      deadbeef, M, 0, N, N, N, 1, deadbeef, x, 0, N, 1, deadbeef, out, 0, N, 1);
  gettimeofday(&stop, NULL);
  check_matvec_adjoint(M, ans.dM.aligned, x, ans.dx.aligned, dout,
                       "MLIR full dense");
  free(ans.dM.aligned);
  free(ans.dx.aligned);
  return timediff(start, stop);
}

unsigned long mlir_tri_primal(double *M, double *dM, double *icf, double *dicf,
                              double *x, double *dx, double *out,
                              double *dout) {
  for (size_t run = 0; run < NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    for (size_t i = 0; i < N; i++) {
      out[i] = 0;
    }

    F64Descriptor1D ans =
        mlir_trimatvec_tri_primal(deadbeef, M, 0, N, N, N, 1, deadbeef, x, 0, N,
                                  1, deadbeef, out, 0, N, 1);
    gettimeofday(&stop, NULL);
    check_matvec_primal(M, x, ans.aligned, "MLIR full tri");
    return timediff(start, stop);
  }
}
unsigned long mlir_tri_adjoint(double *M, double *dM, double *icf, double *dicf,
                               double *x, double *dx, double *out,
                               double *dout) {}
unsigned long mlir_compressed_primal(double *M, double *dM, double *icf,
                                     double *dicf, double *x, double *dx,
                                     double *out, double *dout) {}
unsigned long mlir_compressed_adjoint(double *M, double *dM, double *icf,
                                      double *dicf, double *x, double *dx,
                                      double *out, double *dout) {}
int main() {
  double *M = (double *)malloc(N * N * sizeof(double));
  double *dM = (double *)malloc(N * N * sizeof(double));
  double tri_size = N * (N - 1) / 2;
  double *icf = (double *)malloc(tri_size * sizeof(double));
  double *dicf = (double *)malloc(tri_size * sizeof(double));
  double *x = (double *)malloc(N * sizeof(double));
  double *dx = (double *)malloc(N * sizeof(double));
  double *out = (double *)malloc(N * sizeof(double));
  double *dout = (double *)malloc(N * sizeof(double));
  double *deadbeef = (double *)0xdeadbeef;
  for (size_t i = 0; i < N; i++) {
    dout[i] = 1.0;
  }

  bodyFunc funcs[12] = {
      enzyme_dense_primal, enzyme_dense_adjoint,     enzyme_tri_primal,
      enzyme_tri_adjoint,  enzyme_compressed_primal, enzyme_compressed_adjoint,
      mlir_dense_primal,   mlir_dense_adjoint,       mlir_tri_primal,
      mlir_tri_adjoint,    mlir_compressed_primal,   mlir_compressed_adjoint};

  random_init_d_2d(M, N, N);
  make_ltri(M, N, N);
  convert_to_compressed(M, icf);
  random_init_d_2d(x, N, 1);

  // mlir vs enzyme (2), dense vs tri vs compressed (3), adjoint vs primal (2)
  unsigned long *results_df =
      (unsigned long *)malloc(12 * NUM_RUNS * sizeof(unsigned long));

  for (size_t app = 0; app < 12; app++) {
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[app * NUM_RUNS + run] =
          (*funcs[app])(M, dM, icf, dicf, x, dx, out, dout);
    }
  }
  printf("first result: %lu\n", results_df[0]);

  unsigned long enzyme_primal_full_dense_primal_results[NUM_RUNS];
  unsigned long enzyme_primal_full_dense_adjoint_results[NUM_RUNS];
  unsigned long enzyme_primal_full_tri_primal_results[NUM_RUNS];
  unsigned long enzyme_primal_full_tri_adjoint_results[NUM_RUNS];
  unsigned long enzyme_primal_compressed_primal_results[NUM_RUNS];
  unsigned long enzyme_primal_compressed_adjoint_results[NUM_RUNS];
  unsigned long mlir_primal_full_dense_results[NUM_RUNS];
  unsigned long mlir_adjoint_full_dense_results[NUM_RUNS];
  unsigned long mlir_primal_full_tri_results[NUM_RUNS];
  unsigned long mlir_adjoint_full_tri_results[NUM_RUNS];

  // Enzyme full materialization and dense computation.
  // Primal
  for (size_t run = 0; run < NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    enzyme_trimatvec_dense_primal(M, x, out, N);
    gettimeofday(&stop, NULL);
    enzyme_primal_full_dense_primal_results[run] = timediff(start, stop);
    check_matvec_primal(M, x, out, "enzyme full dense");
  }

  // Adjoint
  for (size_t run = 0; run < NUM_RUNS; run++) {
    struct timeval start, stop;
    size_t M_size = N * N;
    gettimeofday(&start, NULL);
    for (size_t i = 0; i < M_size; i++) {
      dM[i] = 0;
    }
    for (size_t i = 0; i < N; i++) {
      dx[i] = 0;
    }

    for (size_t i = 0; i < N; i++) {
      out[i] = 0;
    }
    enzyme_trimatvec_dense_adjoint(M, dM, x, dx, out, dout, N);
    gettimeofday(&stop, NULL);
    enzyme_primal_full_dense_adjoint_results[run] = timediff(start, stop);
    check_matvec_adjoint(M, dM, x, dx, dout, "enzyme full dense");
  }

  // Enzyme full materialization and triangular computation.
  // Primal
  for (size_t run = 0; run < NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    enzyme_trimatvec_tri_primal(M, x, out, N);
    gettimeofday(&stop, NULL);
    enzyme_primal_full_tri_primal_results[run] = timediff(start, stop);
    check_matvec_primal(M, x, out, "enzyme full tri");
  }

  // Adjoint
  for (size_t run = 0; run < NUM_RUNS; run++) {
    struct timeval start, stop;
    size_t M_size = N * N;
    gettimeofday(&start, NULL);
    for (size_t i = 0; i < M_size; i++) {
      dM[i] = 0;
    }
    for (size_t i = 0; i < N; i++) {
      dx[i] = 0;
    }
    enzyme_trimatvec_tri_adjoint(M, dM, x, dx, out, dout, N);
    gettimeofday(&stop, NULL);
    enzyme_primal_full_tri_adjoint_results[run] = timediff(start, stop);
    check_matvec_adjoint(M, dM, x, dx, dout, "enzyme full tri");
  }

  // Enzyme compressed materialization
  // Primal
  for (size_t run = 0; run < NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    enzyme_trimatvec_compressed_primal(icf, x, out, N);
    gettimeofday(&stop, NULL);
    enzyme_primal_compressed_primal_results[run] = timediff(start, stop);
    check_matvec_primal(M, x, out, "enzyme compressed");
  }

  // Adjoint
  for (size_t run = 0; run < NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    for (size_t i = 0; i < tri_size; i++) {
      dicf[i] = 0;
    }
    for (size_t i = 0; i < N; i++) {
      dx[i] = 0;
    }
    enzyme_trimatvec_compressed_adjoint(icf, dicf, x, dx, out, dout, N);
    gettimeofday(&stop, NULL);
    enzyme_primal_compressed_adjoint_results[run] = timediff(start, stop);
    check_matvec_adjoint(M, dM, x, dx, dout, "enzyme compressed");
  }

  // MLIR full materialization and dense computation
  // Primal
  for (size_t run = 0; run < NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    for (size_t i = 0; i < N; i++) {
      out[i] = 0;
    }

    F64Descriptor1D ans =
        mlir_trimatvec_dense_primal(deadbeef, M, 0, N, N, N, 1, deadbeef, x, 0,
                                    N, 1, deadbeef, out, 0, N, 1);
    gettimeofday(&stop, NULL);
    mlir_primal_full_dense_results[run] = timediff(start, stop);
    check_matvec_primal(M, x, ans.aligned, "MLIR full dense");
    free(ans.aligned);
  }

  // Adjoint
  for (size_t run = 0; run < NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    for (size_t i = 0; i < N; i++) {
      out[i] = 0;
    }

    F64MatVecGradient ans =
        mlir_trimatvec_dense_adjoint(deadbeef, M, 0, N, N, N, 1, deadbeef, x, 0,
                                     N, 1, deadbeef, out, 0, N, 1);
    gettimeofday(&stop, NULL);

    mlir_adjoint_full_dense_results[run] = timediff(start, stop);
    check_matvec_adjoint(M, ans.dM.aligned, x, ans.dx.aligned, dout,
                         "MLIR full dense");
    free(ans.dM.aligned);
    free(ans.dx.aligned);
  }

  // MLIR full materialization and triangular computation
  // Primal
  for (size_t run = 0; run < NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    for (size_t i = 0; i < N; i++) {
      out[i] = 0;
    }

    F64Descriptor1D ans =
        mlir_trimatvec_tri_primal(deadbeef, M, 0, N, N, N, 1, deadbeef, x, 0, N,
                                  1, deadbeef, out, 0, N, 1);
    gettimeofday(&stop, NULL);
    mlir_primal_full_tri_results[run] = timediff(start, stop);
    check_matvec_primal(M, x, ans.aligned, "MLIR full tri");
  }

  // Adjoint
  for (size_t run = 0; run < NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    for (size_t i = 0; i < N; i++) {
      out[i] = 0;
    }

    F64MatVecGradient ans =
        mlir_trimatvec_tri_adjoint(deadbeef, M, 0, N, N, N, 1, deadbeef, x, 0,
                                   N, 1, deadbeef, out, 0, N, 1);
    gettimeofday(&stop, NULL);

    mlir_adjoint_full_tri_results[run] = timediff(start, stop);
    check_matvec_adjoint(M, ans.dM.aligned, x, ans.dx.aligned, dout,
                         "MLIR full tri");
    free(ans.dx.aligned);
  }

  // printf("***Enzyme***\n");
  // printf("Full mat dense computation:\n");
  // printf("Primal:\n");
  // printf("%f\n", take_avg(enzyme_primal_full_dense_primal_results +
  // NUM_WARMUPS,
  //                         NUM_RUNS - NUM_WARMUPS));
  // // print_ul_arr(enzyme_primal_full_dense_primal_results, NUM_RUNS);
  // printf("Adjoint:\n");
  // printf("%f\n",
  //        take_avg(enzyme_primal_full_dense_adjoint_results + NUM_WARMUPS,
  //                 NUM_RUNS - NUM_WARMUPS));
  // // print_ul_arr(enzyme_primal_full_dense_adjoint_results, NUM_RUNS);
  // printf("Full mat tri computation:\n");
  // printf("Primal:\n");
  // printf("%f\n", take_avg(enzyme_primal_full_tri_primal_results +
  // NUM_WARMUPS,
  //                         NUM_RUNS - NUM_WARMUPS));
  // // print_ul_arr(enzyme_primal_full_tri_primal_results, NUM_RUNS);
  // printf("Adjoint:\n");
  // printf("%f\n", take_avg(enzyme_primal_full_tri_adjoint_results +
  // NUM_WARMUPS,
  //                         NUM_RUNS - NUM_WARMUPS));
  // // print_ul_arr(enzyme_primal_full_tri_adjoint_results, NUM_RUNS);
  // printf("Compressed computation:\n");
  // printf("Primal:\n");
  // printf("%f\n", take_avg(enzyme_primal_compressed_primal_results +
  // NUM_WARMUPS,
  //                         NUM_RUNS - NUM_WARMUPS));
  // // print_ul_arr(enzyme_primal_compressed_primal_results, NUM_RUNS);
  // printf("Adjoint:\n");
  // printf("%f\n",
  //        take_avg(enzyme_primal_compressed_adjoint_results + NUM_WARMUPS,
  //                 NUM_RUNS - NUM_WARMUPS));
  // // print_ul_arr(enzyme_primal_compressed_adjoint_results, NUM_RUNS);

  // printf("***MLIR***\n");
  // printf("Full mat dense computation:\n");
  // printf("Primal:\n");
  // printf("%f\n", take_avg(mlir_primal_full_dense_results + NUM_WARMUPS,
  //                         NUM_RUNS - NUM_WARMUPS));
  // printf("Adjoint:\n");
  // printf("%f\n", take_avg(mlir_adjoint_full_dense_results + NUM_WARMUPS,
  //                         NUM_RUNS - NUM_WARMUPS));
  // printf("Full mat tri computation:\n");
  // printf("Primal:\n");
  // printf("%f\n", take_avg(mlir_primal_full_tri_results + NUM_WARMUPS,
  //                         NUM_RUNS - NUM_WARMUPS));
  // printf("Adjoint:\n");
  // printf("%f\n", take_avg(mlir_adjoint_full_tri_results + NUM_WARMUPS,
  //                         NUM_RUNS - NUM_WARMUPS));
  // Need to compute primals and adjoints for dense vs triangular, full vs
  // compressed
  free(M);
  free(dM);
  free(icf);
  free(dicf);
  free(x);
  free(dx);
  free(out);
  free(dout);
}
