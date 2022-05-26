#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUM_RUNS 6
// #define N {{n}}
#define N 1024

double *deadbeef = (double *)0xdeadbeef;
extern F64Descriptor1D lagrad_dot(/*arg0=*/double *, double *, int64_t, int64_t,
                                  int64_t,
                                  /*arg1=*/double *, double *, int64_t, int64_t,
                                  int64_t);
extern double *enzyme_c_dot(int64_t, double *, double *);

void check_dot_err(double *dx, double *y, const char *application) {
  double err = 0.0;
  for (size_t i = 0; i < N; i++) {
    err += fabs(dx[i] - y[i]);
  }

  if (err > 1e-6) {
    printf("(%s) error: %f\n", application, err);
  }
}

typedef unsigned long (*dotBodyFunc)(double *x, double *y);

unsigned long collect_lagrad_dot(double *x, double *y) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor1D res = lagrad_dot(deadbeef, x, 0, N, 1, deadbeef, y, 0, N, 1);
  gettimeofday(&stop, NULL);

  check_dot_err(res.aligned, y, "LAGrad Dot Product");
  free(res.aligned);
  return timediff(start, stop);
}

unsigned long collect_enzyme_c_dot(double *x, double *y) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double *res = enzyme_c_dot(N, x, y);
  gettimeofday(&stop, NULL);

  check_dot_err(res, y, "Enzyme/C Dot Product");
  free(res);
  return timediff(start, stop);
}

int dot_main() {
  double *x = malloc(N * sizeof(double));
  double *y = malloc(N * sizeof(double));
  random_init_d_2d(x, N, 1);
  random_init_d_2d(y, N, 1);

  dotBodyFunc funcs[] = {collect_enzyme_c_dot, collect_lagrad_dot};
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);
  for (size_t app = 0; app < num_apps; app++) {
    unsigned long results_df[NUM_RUNS];
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = (*funcs[app])(x, y);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free(x);
  free(y);
  return 0;
}

extern F64Descriptor1D lagrad_arrmax(double *, double *, int64_t, int64_t,
                                     int64_t);
extern double *enzyme_c_arrmax(int64_t, double *);

void check_arrmax(double *res, const char *application) {
  double err = 0;
  for (size_t i = 0; i < N; i++) {
    err += fabs(res[i] - ((i == 3) ? 1.0 : 0.0));
  }
  if (err > 1e-6) {
    printf("(%s) Arrmax Error: %f\n", application, err);
  }
}

typedef unsigned long (*arrmaxBodyFunc)(double *);
unsigned long collect_lagrad_arrmax(double *arr) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor1D res = lagrad_arrmax(deadbeef, arr, 0, 4, 1);
  gettimeofday(&stop, NULL);
  check_arrmax(res.aligned, "LAGrad");
  free(res.aligned);
  return timediff(start, stop);
}

unsigned long collect_enzyme_c_arrmax(double *arr) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double *enzyme_res = enzyme_c_arrmax(N, arr);
  gettimeofday(&stop, NULL);
  check_arrmax(enzyme_res, "Enzyme/C");
  free(enzyme_res);
  return timediff(start, stop);
}

int arrmax_main() {
  double *arr = malloc(N * sizeof(double));
  for (size_t i = 0; i < N; i++) {
    arr[i] = (i == 3) ? 100.0 : 1.0;
  }
  arrmaxBodyFunc funcs[] = {collect_lagrad_arrmax, collect_enzyme_c_arrmax};
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);
  for (size_t app = 0; app < num_apps; app++) {
    unsigned long results_df[NUM_RUNS];
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = (*funcs[app])(arr);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }
  free(arr);
  return 0;
}

extern F64Descriptor1D
lagrad_vecadd(/*x=*/double *, double *, int64_t, int64_t, int64_t,
              /*y=*/double *, double *, int64_t, int64_t, int64_t,
              /*g=*/double *, double *, int64_t, int64_t, int64_t);
extern double *enzyme_c_vecadd(int64_t, double *, double *, double *);

void check_identity(double *jacobian, const char *application) {
  double err = 0.0;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      err += fabs((i == j) ? (1.0 - jacobian[i * N + j]) : jacobian[i * N + j]);
    }
  }
  if (err > 1e-8) {
    printf("(%s) vector add error: %f\n", application, err);
  }
}

typedef unsigned long (*vecaddBodyFunc)(double *x, double *y, double *g,
                                        double *jacobian);
unsigned long collect_lagrad_vecadd(double *x, double *y, double *g,
                                    double *jacobian) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      g[j] = (i == j) ? 1.0 : 0.0;
    }

    F64Descriptor1D res = lagrad_vecadd(deadbeef, x, 0, N, 1, deadbeef, y, 0, N,
                                        1, deadbeef, g, 0, N, 1);
    for (size_t j = 0; j < N; j++) {
      jacobian[i * N + j] = res.aligned[j];
    }
    // The LAGrad implementation doesn't allocate its own memory, so it doesn't
    // need to be freed.
  }
  gettimeofday(&stop, NULL);
  check_identity(jacobian, "LAGrad");
  return timediff(start, stop);
}

unsigned long collect_enzyme_c_vecadd(double *x, double *y, double *g,
                                      double *jacobian) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      g[j] = (i == j) ? 1.0 : 0.0;
    }

    double *res = enzyme_c_vecadd(N, x, y, g);
    for (size_t j = 0; j < N; j++) {
      jacobian[i * N + j] = res[j];
    }
    free(res);
  }
  gettimeofday(&stop, NULL);
  check_identity(jacobian, "Enzyme/C");
  return timediff(start, stop);
}

int main() {
  double *x = malloc(N * sizeof(double));
  double *y = malloc(N * sizeof(double));
  double *g = malloc(N * sizeof(double));
  double *jacobian = malloc(N * N * sizeof(double));
  random_init_d_2d(x, N, 1);
  random_init_d_2d(y, N, 1);

  vecaddBodyFunc funcs[] = {collect_lagrad_vecadd, collect_enzyme_c_vecadd};
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);
  for (size_t app = 0; app < num_apps; app++) {
    unsigned long results_df[NUM_RUNS];
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = (*funcs[app])(x, y, g, jacobian);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free(x);
  free(y);
  free(g);
  free(jacobian);
}
