#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// #include <sys/time.h>

#define NUM_RUNS 6
double *deadbeef = (double *)0xdeadbeef;

typedef struct {
  double *allocated;
  double *aligned;
  int64_t offset;
  int64_t size_0;
  int64_t size_1;
  int64_t stride_0;
  int64_t stride_1;
} F64Descriptor2D;

extern F64Descriptor2D lagrad_matmul(/*A=*/double *, double *, int64_t, int64_t,
                                     int64_t, int64_t, int64_t,
                                     /*B=*/double *, double *, int64_t, int64_t,
                                     int64_t, int64_t, int64_t);
extern F64Descriptor2D enzyme_mlir_matmul(/*A=*/double *, double *, int64_t,
                                          int64_t, int64_t, int64_t, int64_t,
                                          /*B=*/double *, double *, int64_t,
                                          int64_t, int64_t, int64_t, int64_t);
extern double *enzyme_c_matmul(int64_t, double *, double *);

void check_matmul_grad(size_t N, double *res, double *B,
                       const char *application) {
  double max_err = 0.0;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      double err = fabs(res[i * N + j] - N * B[j * N + i]);
      if (err > max_err) {
        max_err = err;
      }
    }
  }
  if (max_err > 1e-7) {
    printf("(%s) matmul error: %.4e\n", application, max_err);
  }
}

typedef unsigned long (*matmulBodyFunc)(size_t N, double *, double *);

unsigned long collect_lagrad_matmul(size_t N, double *A, double *B) {
  // struct timeval start, stop;
  clock_t start, stop;
  // gettimeofday(&start, NULL);
  start = clock();
  F64Descriptor2D res =
      lagrad_matmul(deadbeef, A, 0, N, N, N, 1, deadbeef, B, 0, N, N, N, 1);
  stop = clock();
  // gettimeofday(&stop, NULL);
  check_matmul_grad(N, res.aligned, B, "LAGrad");
  free(res.aligned);
  return stop - start;
}

unsigned long collect_enzyme_mlir_matmul(size_t N, double *A, double *B) {
  // struct timeval start, stop;
  // gettimeofday(&start, NULL);
  clock_t start, stop;
  start = clock();
  F64Descriptor2D res = enzyme_mlir_matmul(deadbeef, A, 0, N, N, N, 1, deadbeef,
                                           B, 0, N, N, N, 1);
  stop = clock();
  // gettimeofday(&stop, NULL);
  check_matmul_grad(N, res.aligned, B, "Enzyme/MLIR");
  free(res.aligned);
  return stop - start;
}

unsigned long collect_enzyme_c_matmul(size_t N, double *A, double *B) {
  clock_t start, stop;
  // struct timeval start, stop;
  // gettimeofday(&start, NULL);
  start = clock();
  double *res = enzyme_c_matmul(N, A, B);
  // gettimeofday(&stop, NULL);
  stop = clock();
  check_matmul_grad(N, res, B, "Enzyme/C");
  free(res);
  // return timediff(start, stop);
  return stop - start;
}

void print_ul_arr(unsigned long *arr, size_t n) {
  printf("[");
  for (size_t i = 0; i < n; i++) {
    printf("%lu", arr[i]);
    if (i != n - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

typedef struct MatmulApp {
  const char *name;
  unsigned long (*func)(size_t, double *, double *);
} MatmulApp;

int main() {
  // size_t N = strtoul("{{n}}", NULL, 10);
  size_t N = 256;
  double *A = malloc(N * N * sizeof(double));
  double *B = malloc(N * N * sizeof(double));
  for (size_t i = 0; i < N * N; i++) {
    A[i] = 1.4;
    B[i] = 1.2;
  }

  MatmulApp apps[] = {
      {.name = "Enzyme/MLIR", .func = collect_enzyme_mlir_matmul},
      {.name = "LAGrad", .func = collect_lagrad_matmul},
      {.name = "Enzyme/C", .func = collect_enzyme_c_matmul}};

  // matmulBodyFunc funcs[] = {collect_lagrad_matmul,
  //                           // collect_enzyme_mlir_matmul,
  //                           collect_enzyme_c_matmul};
  size_t num_apps = sizeof(apps) / sizeof(apps[0]);
  unsigned long results_df[NUM_RUNS];
  for (size_t app = 0; app < num_apps; app++) {
    printf("%s: ", apps[app].name);
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = apps[app].func(N, A, B);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free(A);
  free(B);
}
