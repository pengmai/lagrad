#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUM_RUNS 6
#define N {{n}}
// #define N 1024
double *deadbeef = (double *)0xdeadbeef;

extern double lagrad_vecscal(
    /*arr=*/double *, double *, int64_t, int64_t, int64_t, /*scal=*/double,
    /*g=*/double *, double *, int64_t, int64_t, int64_t);
extern double enzyme_c_vecscal(int64_t, double, double *, double *);
extern double enzyme_mlir_vecscal(/*arr=*/double *, double *, int64_t, int64_t,
                                  int64_t,
                                  /*scal=*/double *, double *, int64_t,
                                  /*g*/ double *, double *, int64_t, int64_t,
                                  int64_t);

typedef unsigned long (*vecscalBodyFunc)(double *arr, double scal,
                                         double *jacobian, double *g);

void check_eq(double *dx, double *y, const char *application) {
  double err = 0.0;
  for (size_t i = 0; i < N; i++) {
    err += fabs(dx[i] - y[i]);
  }

  if (err > 1e-6) {
    printf("(%s) vecscal error: %f\n", application, err);
  }
}

unsigned long collect_lagrad_vecscal(double *arr, double scal, double *jacobian,
                                     double *g) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      g[j] = (i == j) ? 1.0 : 0.0;
    }

    jacobian[i] =
        lagrad_vecscal(deadbeef, arr, 0, N, 1, scal, deadbeef, g, 0, N, 1);
  }

  gettimeofday(&stop, NULL);
  check_eq(jacobian, arr, "LAGrad");
  return timediff(start, stop);
}

unsigned long collect_enzyme_mlir_vecscal(double *arr, double scal,
                                          double *jacobian, double *g) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      g[j] = (i == j) ? 1.0 : 0.0;
    }
    jacobian[i] = enzyme_mlir_vecscal(deadbeef, arr, 0, N, 1, deadbeef, &scal,
                                      0, deadbeef, g, 0, N, 1);
  }
  gettimeofday(&stop, NULL);
  check_eq(jacobian, arr, "Enzyme/MLIR");
  return timediff(start, stop);
}

unsigned long collect_enzyme_c_vecscal(double *arr, double scal,
                                       double *jacobian, double *g) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      g[j] = (i == j) ? 1.0 : 0.0;
    }

    jacobian[i] = enzyme_c_vecscal(N, scal, arr, g);
  }

  gettimeofday(&stop, NULL);
  check_eq(jacobian, arr, "Enzyme/C");
  return timediff(start, stop);
}

int main() {
  double *arr = malloc(N * sizeof(double));
  double scal = 1.343;
  random_init_d_2d(arr, N, 1);

  double *g = malloc(N * sizeof(double));
  double *jacobian = malloc(N * sizeof(double));
  vecscalBodyFunc funcs[] = {collect_lagrad_vecscal,
                             collect_enzyme_mlir_vecscal,
                             collect_enzyme_c_vecscal};
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);
  for (size_t app = 0; app < num_apps; app++) {
    unsigned long results_df[NUM_RUNS];
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = (*funcs[app])(arr, scal, jacobian, g);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free(g);
  free(jacobian);
  free(arr);
  return 0;
}
