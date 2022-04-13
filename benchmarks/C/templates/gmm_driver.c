#include "gmm.h"
#include "mlir_c_abi.h"
#include "shared_types.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define NUM_RUNS 3

typedef struct {
  double *data;
  double *aligned;
} UnrankedMemref;

typedef struct {
  double *dalphas, *dmeans, *dQs, *dLs, *dicf;
} GradPointers;

double *deadbeef = (double *)0xdeadbeef;

extern double mlir_gmm_opt_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

extern UnrankedMemref lagrad_gmm_objective_tri(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

extern double mlir_gmm_opt_compressed(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

extern void lagrad_gmm_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*dalphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*dmeans=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*dQs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*dLs*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
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

extern GMMCompressedGrad lagrad_gmm_compressed(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

extern void enzyme_gmm_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*dalphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*dmeans=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*dQs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*dLs*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

/* Memory-optimized GMM */
extern GMMGrad enzyme_gmm_tri(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*dalphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*dmeans=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*dQs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*dLs*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

extern GMMGrad enzyme_gmm_opt_diff_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

extern void enzyme_gmm_full_primal(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t,
    /*out=*/double *, double *, int64_t);

extern GMMCompressedGrad enzyme_gmm_opt_diff_compressed(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
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
typedef unsigned long (*bodyFunc)(GMMInput gmm_input, GradPointers gradPointers,
                                  double *ref_alphas, double *ref_means,
                                  double *ref_icf, double *temp_icf);

unsigned long
collect_enzyme_C_compressed_adjoint(GMMInput gmm_input, GradPointers ptrs,
                                    double *ref_alphas, double *ref_means,
                                    double *ref_icf, double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;

  struct timeval start, stop;
  double err = 0.0, errb = 1.0;
  uniform_init_d(0, ptrs.dalphas, k);
  uniform_init_d(0, ptrs.dmeans, d * k);
  uniform_init_d(0, ptrs.dicf, k * (d * (d + 1) / 2));

  gettimeofday(&start, NULL);
  dgmm_objective(&gmm_input, ptrs.dalphas, ptrs.dmeans, ptrs.dicf, &err, &errb);
  gettimeofday(&stop, NULL);

  check_gmm_err(d, k, n, ptrs.dalphas, ref_alphas, ptrs.dmeans, ref_means,
                ptrs.dicf, ref_icf, "Enzyme/C Compressed");

  return timediff(start, stop);
}

unsigned long collect_enzyme_C_full_adjoint(GMMInput gmm_input,
                                            GradPointers ptrs,
                                            double *ref_alphas,
                                            double *ref_means, double *ref_icf,
                                            double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;

  struct timeval start, stop;
  double err = 0.0, errb = 1.0;
  uniform_init_d(0, ptrs.dalphas, k);
  uniform_init_d(0, ptrs.dmeans, d * k);
  uniform_init_d(0, ptrs.dQs, d * k);
  uniform_init_d(0, ptrs.dLs, k * d * d);

  gettimeofday(&start, NULL);
  dgmm_objective_full_L(&gmm_input, ptrs.dalphas, ptrs.dmeans, ptrs.dQs,
                        ptrs.dLs, &err, &errb);
  gettimeofday(&stop, NULL);

  convert_ql_to_icf(d, k, n, ptrs.dQs, ptrs.dLs, temp_icf);
  check_gmm_err(d, k, n, ptrs.dalphas, ref_alphas, ptrs.dmeans, ref_means,
                temp_icf, ref_icf, "Enzyme/C Full");

  return timediff(start, stop);
}

// unsigned long collect_enzyme_full_adjoint(GMMInput gmm_input,
//                                           double *ref_alphas, double
//                                           *ref_means, double *ref_icf, double
//                                           *temp_icf) {
//   int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
//   double *dalphas = malloc(k * sizeof(double));
//   uniform_init_d(0, dalphas, k);
//   double *dmeans = malloc(k * d * sizeof(double));
//   uniform_init_d(0, dmeans, k * d);
//   double *dQs = malloc(k * d * sizeof(double));
//   uniform_init_d(0, dQs, k * d);
//   double *dLs = malloc(k * d * d * sizeof(double));
//   uniform_init_d(0, dLs, k * d * d);
//   struct timeval start, stop;

//   gettimeofday(&start, NULL);
//   enzyme_gmm_full(
//       /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1,
//       /*dalphas=*/deadbeef, dalphas, 0, k, 1,
//       /*means=*/deadbeef, gmm_input.means, 0, k, d, d, 1,
//       /*dmeans=*/deadbeef, dmeans, 0, k, d, d, 1,
//       /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d, d, 1,
//       /*dQs=*/deadbeef, dQs, 0, k, d, d, 1,
//       /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
//       /*dLs=*/deadbeef, dLs, 0, k, d, d, d * d, d, 1,
//       /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
//       /*wishart_gamma=*/gmm_input.wishart_gamma,
//       /*wishart_m=*/gmm_input.wishart_m);
//   gettimeofday(&stop, NULL);

//   printf("Enzyme dmeans:\n");
//   print_d_arr(dmeans, 10);
//   convert_ql_to_icf(d, k, n, dQs, dLs, temp_icf);
//   check_gmm_err(d, k, n, dalphas, ref_alphas, dmeans, ref_means, temp_icf,
//                 ref_icf, "Enzyme (memory optimized) Full");
//   free(dalphas);
//   free(dmeans);
//   free(dQs);
//   free(dLs);
//   return timediff(start, stop);
// }

// unsigned long collect_enzyme_full_primal(GMMInput gmm_input, double
// *ref_alphas,
//                                          double *ref_means, double *ref_icf,
//                                          double *temp_icf) {
//   int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
//   struct timeval start, stop;
//   gettimeofday(&start, NULL);
//   double res;
//   enzyme_gmm_full_primal(
//       /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
//       gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d,
//       d, 1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
//       /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
//       /*wishart_gamma=*/gmm_input.wishart_gamma,
//       /*wishart_m=*/gmm_input.wishart_m, /*out=*/deadbeef, &res, 0);
//   gettimeofday(&stop, NULL);

//   printf("MLIR opt primal res: %f\n", res);
//   return timediff(start, stop);
// }

unsigned long lagrad_gmm_full_primal(GMMInput gmm_input, double *ref_alphas,
                                     double *ref_means, double *ref_icf,
                                     double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double res = mlir_gmm_opt_full(
      /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
      gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d, d,
      1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
      /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input.wishart_gamma,
      /*wishart_m=*/gmm_input.wishart_m);
  gettimeofday(&stop, NULL);
  printf("MLIR full primal: %f\n", res);
  return timediff(start, stop);
}

unsigned long lagrad_gmm_full_adjoint(GMMInput gmm_input, GradPointers ptrs,
                                      double *ref_alphas, double *ref_means,
                                      double *ref_icf, double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  struct timeval start, stop;
  uniform_init_d(0, ptrs.dalphas, k);
  uniform_init_d(0, ptrs.dmeans, k * d);
  uniform_init_d(0, ptrs.dQs, k * d);
  uniform_init_d(0, ptrs.dLs, k * d * d);

  gettimeofday(&start, NULL);
  lagrad_gmm_full(
      /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1,
      /*dalphas=*/deadbeef, ptrs.dalphas, 0, k, 1,
      /*means=*/deadbeef, gmm_input.means, 0, k, d, d, 1,
      /*dmeans=*/deadbeef, ptrs.dmeans, 0, k, d, d, 1,
      /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d, d, 1,
      /*dQs=*/deadbeef, ptrs.dQs, 0, k, d, d, 1,
      /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
      /*dLs=*/deadbeef, ptrs.dLs, 0, k, d, d, d * d, d, 1,
      /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input.wishart_gamma,
      /*wishart_m=*/gmm_input.wishart_m);
  gettimeofday(&stop, NULL);

  // printf("LAGrad dmeans:\n");
  // print_d_arr(dmeans, 10);
  convert_ql_to_icf(d, k, n, ptrs.dQs, ptrs.dLs, temp_icf);
  check_gmm_err(d, k, n, ptrs.dalphas, ref_alphas, ptrs.dmeans, ref_means,
                temp_icf, ref_icf, "LAGrad Full");
  return timediff(start, stop);
}

unsigned long lagrad_gmm_compressed_adjoint(GMMInput gmm_input,
                                            double *ref_alphas,
                                            double *ref_means, double *ref_icf,
                                            double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  int icf_size = d * (d + 1) / 2;
  int tri_size = d * (d - 1) / 2;
  double *compressed_Ls = (double *)malloc(k * tri_size * sizeof(double));
  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < tri_size; j++) {
      compressed_Ls[i * tri_size + j] = gmm_input.icf[i * icf_size + d + j];
    }
  }
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  GMMCompressedGrad ans = lagrad_gmm_compressed(
      /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
      gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d, d,
      1, /*Ls=*/deadbeef, compressed_Ls, 0, k, tri_size, tri_size, 1,
      /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input.wishart_gamma,
      /*wishart_m=*/gmm_input.wishart_m);
  gettimeofday(&stop, NULL);

  convert_ql_compressed_to_icf(d, k, n, ans.dqs.aligned, ans.dls.aligned,
                               temp_icf);
  check_gmm_err(d, k, n, ans.dalphas.aligned, ref_alphas, ans.dmeans.aligned,
                ref_means, temp_icf, ref_icf, "LAGrad Compressed");
  free(ans.dalphas.aligned);
  free(ans.dmeans.aligned);
  free(ans.dqs.aligned);
  free(ans.dls.aligned);
  free(compressed_Ls);
  return timediff(start, stop);
}

int main() {
  GMMInput gmm_input = read_gmm_data();
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  int icf_size = d * (d + 1) / 2;

  double *dalphas = malloc(k * sizeof(double));
  double *dmeans = malloc(k * d * sizeof(double));
  double *dQs = malloc(k * d * sizeof(double));
  double *dLs = malloc(k * d * d * sizeof(double));
  double *dicf = malloc(k * icf_size * sizeof(double));
  GradPointers gradPointers = {.dalphas = dalphas,
                               .dmeans = dmeans,
                               .dQs = dQs,
                               .dLs = dLs,
                               .dicf = dicf};

  bodyFunc funcs[] = {
      // collect_enzyme_tri_primal,
      // lagrad_gmm_full_primal,
      // collect_enzyme_full_adjoint,
      lagrad_gmm_full_adjoint,
      // collect_enzyme_C_full_adjoint,
      // collect_enzyme_C_compressed_adjoint,
      // lagrad_gmm_tri_primal,
      // lagrad_gmm_tri_adjoint,
      // lagrad_gmm_compressed_primal,
      // lagrad_gmm_compressed_adjoint,
  };
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);

  unsigned long *results_df =
      (unsigned long *)malloc(num_apps * NUM_RUNS * sizeof(unsigned long));

  double *ref_alphas = (double *)malloc(k * sizeof(double));
  double *ref_means = (double *)malloc(d * k * sizeof(double));
  double *ref_icf = (double *)malloc(k * icf_size * sizeof(double));
  double *temp_icf = (double *)malloc(k * icf_size * sizeof(double));

  /* Run this to compute reference results. */
  // serialize_reference(gmm_input);
  // return 0;

  read_gmm_grads(d, k, n, ref_alphas, ref_means, ref_icf);

  for (size_t app = 0; app < num_apps; app++) {
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[app * NUM_RUNS + run] = (*funcs[app])(
          gmm_input, gradPointers, ref_alphas, ref_means, ref_icf, temp_icf);
    }
    print_ul_arr(results_df + app * NUM_RUNS, NUM_RUNS);
  }

  // free_gmm_input(gmm_input);
  free(dalphas);
  free(dmeans);
  free(dQs);
  free(dLs);
  free(dicf);
  free(ref_alphas);
  free(ref_means);
  free(ref_icf);
  free(temp_icf);
  free(results_df);
}
