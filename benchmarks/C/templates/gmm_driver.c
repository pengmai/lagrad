#include "gmm.h"
#include "mlir_c_abi.h"
#include "shared_types.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define NUM_RUNS 100

typedef struct {
  double *data;
  double *aligned;
} UnrankedMemref;
double *deadbeef = (double *)0xdeadbeef;

extern UnrankedMemref lagrad_gmm_objective_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

extern GMMGrad lagrad_gmm_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

extern GMMGrad lagrad_gmm_tri(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

extern void dgmm_objective(GMMInput *gmm, double *alphasb, double *meansb,
                           double *icfb, double *err, double *errb);
extern void dgmm_objective_full_L(GMMInput *gmm, double *alphasb,
                                  double *meansb, double *Qsb, double *Lsb,
                                  double *err, double *errb);

extern double enzyme_gmm_primal(GMMInput gmm);
extern double enzyme_gmm_primal_full(GMMInput gmm);

extern void enzyme_gmm_objective_full_L(int d, int k, int n, double *alphas,
                                        double *means, double *Qs, double *Ls,
                                        double *x, double wishart_gamma,
                                        int wishart_m, double *err);

void free_gmm_input(GMMInput gmm_input) {
  free(gmm_input.alphas);
  free(gmm_input.means);
  free(gmm_input.Qs);
  free(gmm_input.Ls);
  free(gmm_input.x);
}

/* Implementations */
typedef unsigned long (*bodyFunc)(GMMInput gmm_input, double *ref_alphas,
                                  double *ref_means, double *ref_icf,
                                  double *temp_icf);

unsigned long enzyme_gmm_full_primal(GMMInput gmm_input, double *ref_alphas,
                                     double *ref_means, double *ref_icf,
                                     double *temp_icf) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  enzyme_gmm_primal_full(gmm_input);
  gettimeofday(&stop, NULL);
  return timediff(start, stop);
}

unsigned long enzyme_gmm_full_adjoint(GMMInput gmm_input, double *ref_alphas,
                                      double *ref_means, double *ref_icf,
                                      double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;

  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double *alphasb = (double *)malloc(k * sizeof(double));
  double *meansb = (double *)malloc(k * d * sizeof(double));
  double *Qsb = (double *)malloc(k * d * sizeof(double));
  double *Lsb = (double *)malloc(k * d * d * sizeof(double));
  double err = 0.0, errb = 1.0;
  for (size_t i = 0; i < k; i++) {
    alphasb[i] = 0;
  }
  for (size_t i = 0; i < d * k; i++) {
    meansb[i] = 0;
    Qsb[i] = 0;
  }
  for (size_t i = 0; i < k * d * d; i++) {
    Lsb[i] = 0;
  }

  dgmm_objective_full_L(&gmm_input, alphasb, meansb, Qsb, Lsb, &err, &errb);
  gettimeofday(&stop, NULL);
  convert_ql_to_icf(d, k, n, Qsb, Lsb, temp_icf);
  check_gmm_err(d, k, n, alphasb, ref_alphas, meansb, ref_means, temp_icf,
                ref_icf, "Enzyme Full");
  return timediff(start, stop);
}

unsigned long enzyme_gmm_compressed_primal(GMMInput gmm_input,
                                           double *ref_alphas,
                                           double *ref_means, double *ref_icf,
                                           double *temp_icf) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  enzyme_gmm_primal(gmm_input);
  gettimeofday(&stop, NULL);
  return timediff(start, stop);
}

unsigned long enzyme_gmm_compressed_adjoint(GMMInput gmm_input,
                                            double *ref_alphas,
                                            double *ref_means, double *ref_icf,
                                            double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  int icf_size = d * (d + 1) / 2;
  struct timeval start, stop;

  gettimeofday(&start, NULL);
  double *alphasb = (double *)malloc(k * sizeof(double));
  double *meansb = (double *)malloc(d * k * sizeof(double));
  double *icfb = (double *)malloc(icf_size * k * sizeof(double));
  double err = 0.0, errb = 1.0;
  for (size_t i = 0; i < k; i++) {
    alphasb[i] = 0;
  }
  for (size_t i = 0; i < d * k; i++) {
    meansb[i] = 0;
  }
  for (size_t i = 0; i < icf_size * k; i++) {
    icfb[i] = 0;
  }
  dgmm_objective(&gmm_input, alphasb, meansb, icfb, &err, &errb);
  gettimeofday(&stop, NULL);

  check_gmm_err(d, k, n, alphasb, ref_alphas, meansb, ref_means, icfb, ref_icf,
                "Enzyme Compressed");
  free(alphasb);
  free(meansb);
  free(icfb);
  return timediff(start, stop);
}

unsigned long lagrad_gmm_full_primal(GMMInput gmm_input, double *ref_alphas,
                                     double *ref_means, double *ref_icf,
                                     double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  lagrad_gmm_objective_full(
      /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
      gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d, d,
      1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
      /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input.wishart_gamma,
      /*wishart_m=*/gmm_input.wishart_m);
  gettimeofday(&stop, NULL);
  return timediff(start, stop);
}

unsigned long lagrad_gmm_full_adjoint(GMMInput gmm_input, double *ref_alphas,
                                      double *ref_means, double *ref_icf,
                                      double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  GMMGrad ans = lagrad_gmm_full(
      /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
      gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d, d,
      1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
      /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input.wishart_gamma,
      /*wishart_m=*/gmm_input.wishart_m);
  gettimeofday(&stop, NULL);

  convert_ql_to_icf(d, k, n, ans.dqs.aligned, ans.dls.aligned, temp_icf);
  check_gmm_err(d, k, n, ans.dalphas.aligned, ref_alphas, ans.dmeans.aligned,
                ref_means, temp_icf, ref_icf, "LAGrad Full");
  free(ans.dalphas.aligned);
  free(ans.dmeans.aligned);
  free(ans.dqs.aligned);
  free(ans.dls.aligned);
  return timediff(start, stop);
}

unsigned long lagrad_gmm_tri_adjoint(GMMInput gmm_input, double *ref_alphas,
                                     double *ref_means, double *ref_icf,
                                     double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  GMMGrad ans = lagrad_gmm_tri(
      /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
      gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d, d,
      1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
      /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input.wishart_gamma,
      /*wishart_m=*/gmm_input.wishart_m);
  gettimeofday(&stop, NULL);

  convert_ql_to_icf(d, k, n, ans.dqs.aligned, ans.dls.aligned, temp_icf);
  check_gmm_err(d, k, n, ans.dalphas.aligned, ref_alphas, ans.dmeans.aligned,
                ref_means, temp_icf, ref_icf, "LAGrad Full");
  free(ans.dalphas.aligned);
  free(ans.dmeans.aligned);
  free(ans.dqs.aligned);
  free(ans.dls.aligned);
  return timediff(start, stop);
}

int main() {
  GMMInput gmm_input = read_gmm_data();
  int d = gmm_input.d;
  int k = gmm_input.k;
  int n = gmm_input.n;
  int icf_size = d * (d + 1) / 2;

  const int num_apps = 7;
  bodyFunc funcs[num_apps] = {
      enzyme_gmm_full_primal,       enzyme_gmm_full_adjoint,
      enzyme_gmm_compressed_primal, enzyme_gmm_compressed_adjoint,
      lagrad_gmm_full_primal,       lagrad_gmm_full_adjoint,
      lagrad_gmm_tri_adjoint};

  unsigned long *results_df =
      (unsigned long *)malloc(num_apps * NUM_RUNS * sizeof(unsigned long));

  double *ref_alphas = (double *)malloc(k * sizeof(double));
  double *ref_means = (double *)malloc(d * k * sizeof(double));
  double *ref_icf = (double *)malloc(k * icf_size * sizeof(double));
  double *temp_icf = (double *)malloc(k * icf_size * sizeof(double));
  double *DELETEME_Ls = (double *)malloc(k * d * (d - 1) / 2 * sizeof(double));
  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < d * (d - 1) / 2; j++) {
      DELETEME_Ls[i * k + j] = gmm_input.icf[i * icf_size + d + j];
    }
  }
  read_gmm_grads(d, k, n, ref_alphas, ref_means, ref_icf);

  for (size_t app = 0; app < num_apps; app++) {
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[app * NUM_RUNS + run] =
          (*funcs[app])(gmm_input, ref_alphas, ref_means, ref_icf, temp_icf);
    }
    print_ul_arr(results_df + app * NUM_RUNS, NUM_RUNS);
  }

  free_gmm_input(gmm_input);
  free(ref_alphas);
  free(ref_means);
  free(ref_icf);
}
