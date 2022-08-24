#include "memusage.h"
#include "trmv.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

#define NUM_RUNS 6
#define CHECK_MEM false

void random_init(double *arr, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    arr[i] = (double)rand() / (double)RAND_MAX;
  }
}

RunProcDyn rpd;
void check_mem_usage() {
  run_get_dynamic_proc_info(getpid(), &rpd);
  printf("%zu\t%zu\n", rpd.rss, rpd.vsize);
}

void verify_trmv(int N, double *M, double *dM, double *x, double *dx,
                 const char *app) {
  double dM_err = 0.0;
  double dx_err = 0.0;
  double sum;
  int Lidx = 0;
  for (size_t i = 0; i < N; i++) {
    sum = 0;
    for (size_t j = i + 1; j < N; j++) {
      dM_err += fabs(dM[Lidx] - x[i]);
      sum += M[Lidx];
      Lidx++;
    }
    dx_err += fabs(sum - dx[i]);
  }
  if (dM_err > 1e-8) {
    printf("(%s) dM err: %.4e\n", app, dM_err);
  }
  if (dx_err > 1e-8) {
    printf("(%s) dx err: %.4e\n", app, dx_err);
  }

  for (size_t i = 0; i < N; i++) {
    double sum = 0;
    for (size_t j = i; j < N; j++) {
      // sum += M[]
    }
  }
}

typedef TRMVCompressedGrad (*packed_trmv)(int64_t N, double *Mfull, double *x);
unsigned long collect_packed(packed_trmv func, int N, double *M, double *x,
                             const char *name) {
  struct timeval start, stop;

  gettimeofday(&start, NULL);
  TRMVCompressedGrad res = func(N, M, x);
  gettimeofday(&stop, NULL);

  if (CHECK_MEM) {
    check_mem_usage();
  } else {
    verify_trmv(N, M, res.dM.aligned, x, res.dx.aligned, name);
  }
  free(res.dM.aligned);
  free(res.dx.aligned);
  return timediff(start, stop);
}

typedef TRMVGrad (*materialized_trmv)(int64_t N, double *Mfull, double *x);
unsigned long collect_materialized(materialized_trmv func, int N, double *M,
                                   double *x, const char *name) {
  struct timeval start, stop;
  int tri_size = N * (N - 1) / 2;
  double *Mfull = calloc(N * N, sizeof(double));
  expand_ltri(N, M, Mfull);

  gettimeofday(&start, NULL);
  TRMVGrad res = func(N, Mfull, x);
  gettimeofday(&stop, NULL);

  double *dM = malloc(tri_size * sizeof(double));
  collapse_ltri(N, res.dM.aligned, dM);

  if (CHECK_MEM) {
    check_mem_usage();
  } else {
    verify_trmv(N, M, dM, x, res.dx.aligned, name);
  }
  free(dM);
  free(Mfull);
  free(res.dM.aligned);
  free(res.dx.aligned);
  return timediff(start, stop);
}

typedef struct TRMVApp {
  const char *name;
  bool materialized;
  TRMVGrad (*mat_func)(int64_t N, double *Mfull, double *x);
  TRMVCompressedGrad (*pck_func)(int64_t N, double *M, double *x);
} TRMVApp;

int main() {
  int N = strtol("{{n}}", NULL, 10);
  int tri_size = N * (N - 1) / 2;
  double *M = malloc(tri_size * sizeof(double));
  double *x = malloc(N * sizeof(double));
  random_init(M, tri_size);
  random_init(x, N);

  TRMVApp apps[] = {
      //
      {.name = "LAGrad Packed",
       .materialized = false,
       .pck_func = lagrad_trmv_packed_wrapper},
      {.name = "LAGrad Tri",
       .materialized = true,
       .mat_func = lagrad_trmv_tri_wrapper},
      {.name = "LAGrad Full",
       .materialized = true,
       .mat_func = lagrad_trmv_full_wrapper},
      // {.name = "Enzyme Packed",
      //  .materialized = false,
      //  .pck_func = enzyme_trmv_packed_wrapper},
      // {.name = "Enzyme Full",
      //  .materialized = true,
      //  .mat_func = enzyme_trmv_full_wrapper},
      // {.name = "Enzyme Tri",
      //  .materialized = true,
      //  .mat_func = enzyme_trmv_tri_wrapper},
  };
  size_t num_apps = sizeof(apps) / sizeof(apps[0]);
  unsigned long results_df[NUM_RUNS];
  for (size_t app = 0; app < num_apps; app++) {
    printf("%s: ", apps[app].name);
    for (size_t run = 0; run < NUM_RUNS; run++) {
      if (apps[app].materialized) {
        results_df[run] =
            collect_materialized(apps[app].mat_func, N, M, x, apps[app].name);
      } else {
        results_df[run] =
            collect_packed(apps[app].pck_func, N, M, x, apps[app].name);
      }
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free(M);
  free(x);
}
