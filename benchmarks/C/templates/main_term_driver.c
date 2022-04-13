#include "gmm.h"
#include "mlir_c_abi.h"
#include "shared_types.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define N {{n}}
#define K {{k}}
#define D {{d}}
// {% if False %}
#define N 1000
#define K 25
#define D 10
// {% endif %}
#define NUM_RUNS 1

// extern double emain_term(
//     /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
//     /*means*/ double *, double *, int64_t, int64_t, int64_t, int64_t,
//     int64_t,
//     /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
//     /*Ls=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
//     int64_t, int64_t,
//     /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern double main_term(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
// extern GMMGrad main_term_enzyme(
//     /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
//     /*means*/ double *, double *, int64_t, int64_t, int64_t, int64_t,
//     int64_t,
//     /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
//     /*Ls=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
//     int64_t, int64_t,
//     /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern GMMGrad main_term_lagrad(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern double c_main_term(double *alphas, double *means, double *Qs, double *Ls,
                          double *x);
extern void enzyme_c_main_term(double *alphas, double *dalphas, double *means,
                               double *dmeans, double *Qs, double *dQs,
                               double *Ls, double *dLs, double *x);

double *deadbeef = (double *)0xdeadbeef;

typedef struct {
  double *dalphas, *dmeans, *dQs, *dLs;
} GMMRef;

unsigned long collect_enzyme_c(double *alphas, double *means, double *Qs,
                               double *Ls, double *x, GMMRef *ref,
                               bool first_run) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double *dalphas = (double *)malloc(K * sizeof(double));
  double *dmeans = (double *)malloc(K * D * sizeof(double));
  double *dQs = (double *)malloc(K * D * sizeof(double));
  double *dLs = (double *)malloc(K * D * D * sizeof(double));

  uniform_init_d(0, dalphas, K);
  uniform_init_d(0, dmeans, K * D);
  uniform_init_d(0, dQs, K * D);
  uniform_init_d(0, dLs, K * D * D);

  enzyme_c_main_term(alphas, dalphas, means, dmeans, Qs, dQs, Ls, dLs, x);
  gettimeofday(&stop, NULL);

  // printf("Enzyme results:\n");
  // print_d_arr(dLs + 5 * D * D, 10);
  if (first_run) {
    for (size_t i = 0; i < K; i++) {
      ref->dalphas[i] = dalphas[i];
    }
    for (size_t i = 0; i < K * D; i++) {
      ref->dmeans[i] = dmeans[i];
      ref->dQs[i] = dQs[i];
    }
    for (size_t i = 0; i < K * D * D; i++) {
      ref->dLs[i] = dLs[i];
    }
  }

  free(dalphas);
  free(dmeans);
  free(dQs);
  free(dLs);
  return timediff(start, stop);
}

unsigned long collect_lagrad(double *alphas, double *means, double *Qs,
                             double *Ls, double *x, GMMRef *ref,
                             bool first_run) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  GMMGrad lagrad_res =
      main_term_lagrad(deadbeef, alphas, 0, K, 1, deadbeef, means, 0, K, D, D,
                       1, deadbeef, Qs, 0, K, D, D, 1, deadbeef, Ls, 0, K, D, D,
                       D * D, D, 1, deadbeef, x, 0, N, D, D, 1);
  gettimeofday(&stop, NULL);
  // printf("LAGrad results:\n");
  // print_d_arr(lagrad_res.dls.aligned + 5 * D * D, 10);

  if (first_run) {
    bool ok = true;
    double err = 0;
    for (size_t i = 0; i < K; i++) {
      err += fabs(lagrad_res.dalphas.aligned[i] - ref->dalphas[i]);
    }
    if (err > 1e-6) {
      printf("***LAGrad Alphas Err***: %f\n", err);
      ok = false;
    }

    err = 0;
    double err2 = 0;
    for (size_t i = 0; i < K * D; i++) {
      err += fabs(lagrad_res.dmeans.aligned[i] - ref->dmeans[i]);
      err2 += fabs(lagrad_res.dqs.aligned[i] - ref->dQs[i]);
    }
    if (err > 1e-6) {
      printf("***LAGrad Means err***: %f\n", err);
      ok = false;
    }
    if (err2 > 1e-6) {
      printf("***LAGrad Qs err***: %f\n", err2);
      ok = false;
    }

    err = 0;
    for (size_t i = 0; i < K * D * D; i++) {
      err += fabs(lagrad_res.dls.aligned[i] - ref->dLs[i]);
    }
    if (err > 1e-6) {
      printf("***LAGrad Ls err***: %f\n", err);
      ok = false;
    }

    if (ok) {
      printf("LAGrad results are OK.\n");
    }
  }

  free(lagrad_res.dalphas.aligned);
  free(lagrad_res.dmeans.aligned);
  free(lagrad_res.dqs.aligned);
  free(lagrad_res.dls.aligned);
  return timediff(start, stop);
}

int main() {
  GMMInput gmm_input = read_gmm_data();
  double *alphas = (double *)malloc(K * sizeof(double));
  double *means = (double *)malloc(K * D * sizeof(double));
  double *Qs = (double *)malloc(K * D * sizeof(double));
  double *Ls = (double *)malloc(K * D * D * sizeof(double));
  double *x = (double *)malloc(N * D * sizeof(double));

  double *ref_alphas = (double *)malloc(K * sizeof(double));
  double *ref_means = (double *)malloc(K * D * sizeof(double));
  double *ref_Qs = (double *)malloc(K * D * sizeof(double));
  double *ref_Ls = (double *)malloc(K * D * D * sizeof(double));
  GMMRef ref = {
      .dalphas = ref_alphas, .dmeans = ref_means, .dQs = ref_Qs, .dLs = ref_Ls};

  // init_range(alphas, K);
  // init_range(means, K * D);
  // init_range(Qs, K * D);
  // init_range(Ls, K * D * D);
  // init_range(x, N * D);

  // random_init_d_2d(alphas, K, 1);
  // random_init_d_2d(means, K, D);
  // random_init_d_2d(Qs, K, D);
  // random_init_d_2d(Ls, K, D * D);
  // random_init_d_2d(x, N, D);

  // double res = emain_term(deadbeef, alphas, 0, K, 1, deadbeef, means, 0, K,
  // D,
  //                         D, 1, deadbeef, Qs, 0, K, D, D, 1, deadbeef, Ls, 0,
  //                         K, D, D, D * D, D, 1, deadbeef, x, 0, N, D, D, 1);

  // double res = c_main_term(alphas, means, Qs, Ls, x);
  // printf("C main term primal: %f\n", res);
  // double mres = main_term(
  //     deadbeef, gmm_input.alphas, 0, K, 1, deadbeef, gmm_input.means, 0, K,
  //     D, D, 1, deadbeef, gmm_input.Qs, 0, K, D, D, 1, deadbeef, gmm_input.Ls,
  //     0, K, D, D, D * D, D, 1, deadbeef, gmm_input.x, 0, N, D, D, 1);
  // printf("MLIR main term primal: %f\n", mres);

  // collect_enzyme_c(alphas, means, Qs, Ls, x, &ref, true);
  unsigned long results[NUM_RUNS];
  for (size_t run = 0; run < NUM_RUNS; run++) {
    results[run] =
        collect_enzyme_c(gmm_input.alphas, gmm_input.means, gmm_input.Qs,
                         gmm_input.Ls, gmm_input.x, &ref, run == 0);
  }
  print_ul_arr(results, NUM_RUNS);

  // for (size_t run = 0; run < NUM_RUNS; run++) {
  //   results[run] =
  //       collect_lagrad(gmm_input.alphas, gmm_input.means, gmm_input.Qs,
  //                      gmm_input.Ls, gmm_input.x, &ref, run == 0);
  // }
  // print_ul_arr(results, NUM_RUNS);



  // collect_lagrad(gmm_input.alphas, gmm_input.means, gmm_input.Qs,
  // gmm_input.Ls,
  //                gmm_input.x, &ref, false);

  // double primal_diff = fabs(res - mres);
  // if (primal_diff > 1e-6) {
  //   printf("Enzyme primal: %f\nLAGrad primal: %f\n", res, mres);
  // } else {
  //   printf("**Primals are OK**\n");
  // }

  // GMMGrad enzyme_res =
  //     main_term_enzyme(deadbeef, alphas, 0, K, 1, deadbeef, means, 0, K, D,
  //     D,
  //                      1, deadbeef, Qs, 0, K, D, D, 1, deadbeef, Ls, 0, K, D,
  //                      D, D * D, D, 1, deadbeef, x, 0, N, D, D, 1);
  //   unsigned long results[NUM_RUNS];
  //   for (size_t run = 0; run < NUM_RUNS; run++) {
  //   }
  //   print_ul_arr(results, NUM_RUNS);

  free(ref_alphas);
  free(ref_means);
  free(ref_Qs);
  free(ref_Ls);
  free(alphas);
  free(means);
  free(Qs);
  free(Ls);
  free(x);
}