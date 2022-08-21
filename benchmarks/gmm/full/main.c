#include "gmm.h"
#include "gmm_types.h"
#include "lagrad_utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define NUM_RUNS 1

typedef struct {
  double *data;
  double *aligned;
} UnrankedMemref;
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

extern GMMGrad lagrad_gmm_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

GMMGrad lagrad_gmm_full_adjoint(GMMInput *gmm_input) {
  int n = gmm_input->n, k = gmm_input->k, d = gmm_input->d;
  return lagrad_gmm_full(
      /*alphas=*/deadbeef, gmm_input->alphas, 0, k, 1,
      /*means=*/deadbeef, gmm_input->means, 0, k, d, d, 1,
      /*Qs=*/deadbeef, gmm_input->Qs, 0, k, d, d, 1,
      /*Ls=*/deadbeef, gmm_input->Ls, 0, k, d, d, d * d, d, 1,
      /*x=*/deadbeef, gmm_input->x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input->wishart_gamma,
      /*wishart_m=*/gmm_input->wishart_m);
}

extern GMMGrad enzyme_c_gmm_full(GMMInput *gmm);

extern GMMGrad enzyme_gmm_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

/* Memory-optimized GMM */
extern double enzyme_gmm_opt_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
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

extern double enzyme_gmm_opt_compressed(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

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
// unsigned long enzyme_gmm_full_primal(GMMInput gmm_input, double *ref_alphas,
//                                      double *ref_means, double *ref_icf,
//                                      double *temp_icf) {
//   struct timeval start, stop;
//   gettimeofday(&start, NULL);
//   enzyme_gmm_primal_full(gmm_input);
//   gettimeofday(&stop, NULL);
//   return timediff(start, stop);
// }

// unsigned long enzyme_gmm_full_adjoint(GMMInput gmm_input, double *ref_alphas,
//                                       double *ref_means, double *ref_icf,
//                                       double *temp_icf) {
//   int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;

//   struct timeval start, stop;
//   gettimeofday(&start, NULL);
//   double *alphasb = (double *)malloc(k * sizeof(double));
//   double *meansb = (double *)malloc(k * d * sizeof(double));
//   double *Qsb = (double *)malloc(k * d * sizeof(double));
//   double *Lsb = (double *)malloc(k * d * d * sizeof(double));
//   double err = 0.0, errb = 1.0;
//   for (size_t i = 0; i < k; i++) {
//     alphasb[i] = 0;
//   }
//   for (size_t i = 0; i < d * k; i++) {
//     meansb[i] = 0;
//     Qsb[i] = 0;
//   }
//   for (size_t i = 0; i < k * d * d; i++) {
//     Lsb[i] = 0;
//   }

//   dgmm_objective_full_L(&gmm_input, alphasb, meansb, Qsb, Lsb, &err, &errb);
//   gettimeofday(&stop, NULL);
//   convert_ql_to_icf(d, k, n, Qsb, Lsb, temp_icf);
//   check_gmm_err(d, k, n, alphasb, ref_alphas, meansb, ref_means, temp_icf,
//                 ref_icf, "Enzyme Full");
//   return timediff(start, stop);
// }

// unsigned long enzyme_mlir_gmm_full_adjoint(GMMInput gmm_input,
//                                            double *ref_alphas,
//                                            double *ref_means, double
//                                            *ref_icf, double *temp_icf) {
//   int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
//   struct timeval start, stop;
//   gettimeofday(&start, NULL);
//   GMMGrad ans = enzyme_gmm_full(
//       /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
//       gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d,
//       d, 1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
//       /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
//       /*wishart_gamma=*/gmm_input.wishart_gamma,
//       /*wishart_m=*/gmm_input.wishart_m);
//   gettimeofday(&stop, NULL);

//   convert_ql_to_icf(d, k, n, ans.dqs.aligned, ans.dls.aligned, temp_icf);
//   check_gmm_err(d, k, n, ans.dalphas.aligned, ref_alphas, ans.dmeans.aligned,
//                 ref_means, temp_icf, ref_icf, "Enzyme (MLIR Start) Full");
//   free(ans.dalphas.aligned);
//   free(ans.dmeans.aligned);
//   free(ans.dqs.aligned);
//   free(ans.dls.aligned);
//   return timediff(start, stop);
// }

// unsigned long mlir_optimized_full_primal(GMMInput gmm_input, double
// *ref_alphas,
//                                          double *ref_means, double *ref_icf,
//                                          double *temp_icf) {
//   int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
//   struct timeval start, stop;
//   gettimeofday(&start, NULL);
//   enzyme_gmm_opt_full(
//       /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
//       gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d,
//       d, 1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
//       /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
//       /*wishart_gamma=*/gmm_input.wishart_gamma,
//       /*wishart_m=*/gmm_input.wishart_m);
//   gettimeofday(&stop, NULL);
//   return timediff(start, stop);
// }

// unsigned long mlir_optimized_full_adjoint(GMMInput gmm_input,
//                                           double *ref_alphas, double
//                                           *ref_means, double *ref_icf, double
//                                           *temp_icf) {
//   int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
//   struct timeval start, stop;
//   gettimeofday(&start, NULL);
//   GMMGrad ans = enzyme_gmm_opt_diff_full(
//       /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
//       gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d,
//       d, 1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
//       /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
//       /*wishart_gamma=*/gmm_input.wishart_gamma,
//       /*wishart_m=*/gmm_input.wishart_m);
//   gettimeofday(&stop, NULL);

//   // print_d_arr(ans.dls.aligned + 10, 10);
//   convert_ql_to_icf(d, k, n, ans.dqs.aligned, ans.dls.aligned, temp_icf);
//   check_gmm_err(d, k, n, ans.dalphas.aligned, ref_alphas, ans.dmeans.aligned,
//                 ref_means, temp_icf, ref_icf, "Enzyme (memory optimized)
//                 Full");
//   free(ans.dalphas.aligned);
//   free(ans.dmeans.aligned);
//   free(ans.dqs.aligned);
//   free(ans.dls.aligned);
//   return timediff(start, stop);
// }

// unsigned long mlir_optimized_compressed_primal(GMMInput gmm_input,
//                                                double *ref_alphas,
//                                                double *ref_means,
//                                                double *ref_icf,
//                                                double *temp_icf) {
//   int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
//   int tri_size = d * (d - 1) / 2;
//   int icf_size = d * (d + 1) / 2;
//   double *compressed_Ls = (double *)malloc(k * tri_size * sizeof(double));
//   for (size_t i = 0; i < k; i++) {
//     for (size_t j = 0; j < tri_size; j++) {
//       compressed_Ls[i * tri_size + j] = gmm_input.icf[i * icf_size + d + j];
//     }
//   }
//   struct timeval start, stop;
//   gettimeofday(&start, NULL);
//   double res = enzyme_gmm_opt_compressed(
//       /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
//       gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d,
//       d, 1, /*Ls=*/deadbeef, compressed_Ls, 0, k, tri_size, tri_size, 1,
//       /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
//       /*wishart_gamma=*/gmm_input.wishart_gamma,
//       /*wishart_m=*/gmm_input.wishart_m);
//   gettimeofday(&stop, NULL);

//   printf("MLIR opt primal res: %f\n", res);
//   free(compressed_Ls);
//   return timediff(start, stop);
// }

// unsigned long lagrad_gmm_full_primal(GMMInput gmm_input, double *ref_alphas,
//                                      double *ref_means, double *ref_icf,
//                                      double *temp_icf) {
//   int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
//   struct timeval start, stop;
//   gettimeofday(&start, NULL);
//   double res = mlir_gmm_opt_full(
//       /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
//       gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d,
//       d, 1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
//       /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
//       /*wishart_gamma=*/gmm_input.wishart_gamma,
//       /*wishart_m=*/gmm_input.wishart_m);
//   gettimeofday(&stop, NULL);
//   printf("MLIR full primal: %.4e\n", res);
//   return timediff(start, stop);
// }

typedef struct GMMApp {
  const char *name;
  GMMGrad (*func)(GMMInput *gmm_input);
} GMMApp;

unsigned long collect_full_adjoint(GMMApp app, GMMInput *gmm_input,
                                   double *ref_alphas, double *ref_means,
                                   double *ref_icf, double *temp_icf) {
  int d = gmm_input->d, k = gmm_input->k, n = gmm_input->n;
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  GMMGrad ans = app.func(gmm_input);
  gettimeofday(&stop, NULL);

  // print_d_arr_2d(ans.dqs.aligned, ans.dqs.size_0, ans.dqs.size_1);
  // print_d_arr_3d(ans.dls.aligned, ans.dls.size_0, ans.dls.size_1,
  //                ans.dls.size_2);

  // print_d_arr(ans.dls.aligned + 10, 10);
  // convert_ql_compressed_to_icf(d, k, n, ans.dqs.aligned, ans.dls.aligned,
  //                              temp_icf);
  convert_ql_to_icf(d, k, n, ans.dqs.aligned, ans.dls.aligned, temp_icf);
  check_gmm_err(d, k, n, ans.dalphas.aligned, ref_alphas, ans.dmeans.aligned,
                ref_means, temp_icf, ref_icf, app.name);
  free(ans.dalphas.aligned);
  free(ans.dmeans.aligned);
  free(ans.dqs.aligned);
  free(ans.dls.aligned);
  return timediff(start, stop);
}

GMMGrad populate_ref(GMMInput *gmm_input) {
  return lagrad_gmm_full_adjoint(gmm_input);
}

int main() {
  GMMInput gmm_input = read_gmm_data("{{data_file}}");
  int d = gmm_input.d;
  int k = gmm_input.k;
  int n = gmm_input.n;
  printf("d: %d, k: %d, n: %d\n", d, k, n);
  int icf_size = d * (d + 1) / 2;
  GMMApp apps[] = {
      //
      {.name = "LAGrad", .func = lagrad_gmm_full_adjoint},
      {.name = "Enzyme/C", .func = enzyme_c_gmm_full},
  };

  size_t num_apps = sizeof(apps) / sizeof(apps[0]);

  unsigned long results_df[NUM_RUNS];
  double *ref_icf = calloc(k * icf_size, sizeof(double));
  double *temp_icf = calloc(k * icf_size, sizeof(double));
  GMMGrad ref_grad = populate_ref(&gmm_input);
  convert_ql_to_icf(d, k, n, ref_grad.dqs.aligned, ref_grad.dls.aligned,
                    ref_icf);
  free(ref_grad.dqs.aligned);
  free(ref_grad.dls.aligned);

  for (size_t app = 0; app < num_apps; app++) {
    printf("%s: ", apps[app].name);
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] =
          collect_full_adjoint(apps[app], &gmm_input, ref_grad.dalphas.aligned,
                               ref_grad.dmeans.aligned, ref_icf, temp_icf);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free_gmm_input(gmm_input);
  free(ref_grad.dalphas.aligned);
  free(ref_grad.dmeans.aligned);
  free(ref_icf);
  free(temp_icf);
}