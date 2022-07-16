#include "gmm.h"
#include "mlir_c_abi.h"
#include "shared_types.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define NUM_RUNS 4

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

extern GMMCompressedGrad lagrad_gmm_compressed(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

/* Memory-optimized GMM */

extern double enzyme_gmm_opt_compressed(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

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

extern double enzyme_gmm_primal(GMMInput gmm);

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

unsigned long enzyme_mlir_compressed_primal(GMMInput gmm_input,
                                            double *ref_alphas,
                                            double *ref_means, double *ref_icf,
                                            double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  int tri_size = d * (d - 1) / 2;
  int icf_size = d * (d + 1) / 2;
  double *compressed_Ls = (double *)malloc(k * tri_size * sizeof(double));
  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < tri_size; j++) {
      compressed_Ls[i * tri_size + j] = gmm_input.icf[i * icf_size + d + j];
    }
  }
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double res = enzyme_gmm_opt_compressed(
      /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
      gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d, d,
      1, /*Ls=*/deadbeef, compressed_Ls, 0, k, tri_size, tri_size, 1,
      /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input.wishart_gamma,
      /*wishart_m=*/gmm_input.wishart_m);
  gettimeofday(&stop, NULL);

  printf("MLIR opt primal res: %f\n", res);
  free(compressed_Ls);
  return timediff(start, stop);
}

unsigned long enzyme_mlir_compressed_adjoint(GMMInput gmm_input,
                                             double *ref_alphas,
                                             double *ref_means, double *ref_icf,
                                             double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  int tri_size = d * (d - 1) / 2;
  int icf_size = d * (d + 1) / 2;
  double *compressed_Ls = (double *)malloc(k * tri_size * sizeof(double));
  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < tri_size; j++) {
      compressed_Ls[i * tri_size + j] = gmm_input.icf[i * icf_size + d + j];
    }
  }
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  GMMCompressedGrad ans = enzyme_gmm_opt_diff_compressed(
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
                ref_means, temp_icf, ref_icf, "Enzyme/MLIR Compressed");
  free(ans.dalphas.aligned);
  free(ans.dmeans.aligned);
  free(ans.dqs.aligned);
  free(ans.dls.aligned);
  free(compressed_Ls);
  return timediff(start, stop);
}

unsigned long enzyme_gmm_compressed_primal(GMMInput gmm_input,
                                           double *ref_alphas,
                                           double *ref_means, double *ref_icf,
                                           double *temp_icf) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double res = enzyme_gmm_primal(gmm_input);
  gettimeofday(&stop, NULL);
  printf("Enzyme/C Compressed Primal: %f\n", res);
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
                "Enzyme Compressed/C");
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

  // print_d_arr(ans.dls.aligned + 10, 10);
  convert_ql_to_icf(d, k, n, ans.dqs.aligned, ans.dls.aligned, temp_icf);
  check_gmm_err(d, k, n, ans.dalphas.aligned, ref_alphas, ans.dmeans.aligned,
                ref_means, temp_icf, ref_icf, "LAGrad Full");
  free(ans.dalphas.aligned);
  free(ans.dmeans.aligned);
  free(ans.dqs.aligned);
  free(ans.dls.aligned);
  return timediff(start, stop);
}

// unsigned long lagrad_gmm_tri_primal(GMMInput gmm_input, double *ref_alphas,
//                                     double *ref_means, double *ref_icf,
//                                     double *temp_icf) {
//   int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
//   struct timeval start, stop;
//   gettimeofday(&start, NULL);
//   lagrad_gmm_objective_tri(
//       /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
//       gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d,
//       d, 1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
//       /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
//       /*wishart_gamma=*/gmm_input.wishart_gamma,
//       /*wishart_m=*/gmm_input.wishart_m);
//   gettimeofday(&stop, NULL);
//   return timediff(start, stop);
// }

// unsigned long lagrad_gmm_tri_adjoint(GMMInput gmm_input, double *ref_alphas,
//                                      double *ref_means, double *ref_icf,
//                                      double *temp_icf) {
//   int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
//   struct timeval start, stop;
//   gettimeofday(&start, NULL);
//   GMMGrad ans = lagrad_gmm_tri(
//       /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
//       gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d,
//       d, 1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
//       /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
//       /*wishart_gamma=*/gmm_input.wishart_gamma,
//       /*wishart_m=*/gmm_input.wishart_m);
//   gettimeofday(&stop, NULL);

//   convert_ql_to_icf(d, k, n, ans.dqs.aligned, ans.dls.aligned, temp_icf);
//   check_gmm_err(d, k, n, ans.dalphas.aligned, ref_alphas, ans.dmeans.aligned,
//                 ref_means, temp_icf, ref_icf, "LAGrad Tri");
//   free(ans.dalphas.aligned);
//   free(ans.dmeans.aligned);
//   free(ans.dqs.aligned);
//   free(ans.dls.aligned);
//   return timediff(start, stop);
// }

unsigned long lagrad_gmm_compressed_primal(GMMInput gmm_input,
                                           double *ref_alphas,
                                           double *ref_means, double *ref_icf,
                                           double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  int tri_size = d * (d - 1) / 2;
  int icf_size = d * (d + 1) / 2;
  double *compressed_Ls = (double *)malloc(k * tri_size * sizeof(double));
  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < tri_size; j++) {
      compressed_Ls[i * tri_size + j] = gmm_input.icf[i * icf_size + d + j];
    }
  }
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double res = mlir_gmm_opt_compressed(
      /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
      gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d, d,
      1, /*Ls=*/deadbeef, compressed_Ls, 0, k, tri_size, tri_size, 1,
      /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input.wishart_gamma,
      /*wishart_m=*/gmm_input.wishart_m);
  gettimeofday(&stop, NULL);
  printf("MLIR compressed primal: %f\n", res);
  free(compressed_Ls);
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

// void serialize_reference(GMMInput gmm_input) {
//   int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
//   int icf_size = d * (d + 1) / 2;
//   double *alphasb = (double *)malloc(k * sizeof(double));
//   double *meansb = (double *)malloc(d * k * sizeof(double));
//   double *icfb = (double *)malloc(icf_size * k * sizeof(double));
//   double err = 0.0, errb = 1.0;
//   for (size_t i = 0; i < k; i++) {
//     alphasb[i] = 0;
//   }
//   for (size_t i = 0; i < d * k; i++) {
//     meansb[i] = 0;
//   }
//   for (size_t i = 0; i < icf_size * k; i++) {
//     icfb[i] = 0;
//   }
//   dgmm_objective(&gmm_input, alphasb, meansb, icfb, &err, &errb);
//   serialize_gmm_grads("{{results_file}}", d, k, n, alphasb, meansb, icfb);
//   free(alphasb);
//   free(meansb);
//   free(icfb);
// }

void populate_ref(GMMInput gmm_input, double *ref_alphas, double *ref_means,
                  double *ref_icf) {
  int d = gmm_input.d, k = gmm_input.k;
  int icf_size = d * (d + 1) / 2;
  double err = 0.0, errb = 1.0;
  for (size_t i = 0; i < k; i++) {
    ref_alphas[i] = 0;
  }
  for (size_t i = 0; i < d * k; i++) {
    ref_means[i] = 0;
  }
  for (size_t i = 0; i < icf_size * k; i++) {
    ref_icf[i] = 0;
  }
  dgmm_objective(&gmm_input, ref_alphas, ref_means, ref_icf, &err, &errb);
}

int main() {
  GMMInput gmm_input = read_gmm_data("{{data_file}}");
  int d = gmm_input.d;
  int k = gmm_input.k;
  // int n = gmm_input.n;
  int icf_size = d * (d + 1) / 2;

  bodyFunc funcs[] = {
      // enzyme_gmm_full_primal,
      // enzyme_gmm_compressed_primal,

      // enzyme_gmm_compressed_adjoint,

      // enzyme_mlir_gmm_full_adjoint,
      // mlir_optimized_full_primal,
      // enzyme_mlir_full_adjoint,
      // enzyme_c_gmm_full_adjoint,
      // enzyme_gmm_compressed_adjoint,

      // enzyme_mlir_compressed_primal,
      enzyme_mlir_compressed_adjoint,

      // lagrad_gmm_full_primal,
      // lagrad_gmm_full_adjoint,
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
  populate_ref(gmm_input, ref_alphas, ref_means, ref_icf);

  for (size_t app = 0; app < num_apps; app++) {
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[app * NUM_RUNS + run] =
          (*funcs[app])(gmm_input, ref_alphas, ref_means, ref_icf, temp_icf);
    }
    print_ul_arr(results_df + app * NUM_RUNS, NUM_RUNS);
  }

  // free_gmm_input(gmm_input);
  free(ref_alphas);
  free(ref_means);
  free(ref_icf);
  free(temp_icf);
  free(results_df);
}
