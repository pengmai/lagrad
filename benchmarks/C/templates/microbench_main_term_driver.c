#include "gmm.h"
#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUM_RUNS 1
double *deadbeef = (double *)0xdeadbeef;

extern GMMGrad handwritten_main_term_grad(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern GMMCompressedGrad handwritten_main_term_compressed_grad(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern GMMGrad lagrad_main_term(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern GMMCompressedGrad enzyme_mlir_main_term_compressed(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);

typedef unsigned long (*maintermBodyFunc)(GMMInput gmm_input,
                                          double *ref_alphas, double *ref_means,
                                          double *ref_icf, double *temp_icf);

unsigned long collect_handrolled_gradient(GMMInput gmm_input,
                                          double *ref_alphas, double *ref_means,
                                          double *ref_icf, double *temp_icf) {
  int d = gmm_input.d;
  int k = gmm_input.k;
  int n = gmm_input.n;
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  GMMGrad res = handwritten_main_term_grad(
      deadbeef, gmm_input.alphas, 0, k, 1, deadbeef, gmm_input.means, 0, k, d,
      d, 1, deadbeef, gmm_input.Qs, 0, k, d, d, 1, deadbeef, gmm_input.Ls, 0, k,
      d, d, d * d, d, 1, deadbeef, gmm_input.x, 0, n, d, d, 1);
  gettimeofday(&stop, NULL);

  convert_ql_to_icf(d, k, n, res.dqs.aligned, res.dls.aligned, temp_icf);
  check_gmm_err(d, k, n, res.dalphas.aligned, ref_alphas, res.dmeans.aligned,
                ref_means, temp_icf, ref_icf, "Handrolled full");
  free(res.dalphas.aligned);
  free(res.dmeans.aligned);
  free(res.dqs.aligned);
  free(res.dls.aligned);
  return timediff(start, stop);
}

unsigned long collect_handrolled_compressed_gradient(GMMInput gmm_input,
                                                     double *ref_alphas,
                                                     double *ref_means,
                                                     double *ref_icf,
                                                     double *temp_icf) {
  int d = gmm_input.d;
  int k = gmm_input.k;
  int n = gmm_input.n;
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
  GMMCompressedGrad res = handwritten_main_term_compressed_grad(
      deadbeef, gmm_input.alphas, 0, k, 1, deadbeef, gmm_input.means, 0, k, d,
      d, 1, deadbeef, gmm_input.Qs, 0, k, d, d, 1, deadbeef, compressed_Ls, 0,
      k, tri_size, tri_size, 1, deadbeef, gmm_input.x, 0, n, d, d, 1);
  gettimeofday(&stop, NULL);
  convert_ql_compressed_to_icf(d, k, n, res.dqs.aligned, res.dls.aligned,
                               temp_icf);
  check_gmm_err(d, k, n, res.dalphas.aligned, ref_alphas, res.dmeans.aligned,
                ref_means, temp_icf, ref_icf, "Handwritten Compressed");
  // print_d_arr_2d(res.dmeans.aligned, res.dmeans.size_0, res.dmeans.size_1);
  // print_d_arr_2d(res.dqs.aligned, res.dqs.size_0, res.dqs.size_1);
  // print_d_arr_2d(res.dls.aligned, res.dls.size_1, res.dls.size_2);
  free(res.dalphas.aligned);
  free(res.dmeans.aligned);
  free(res.dqs.aligned);
  free(res.dls.aligned);
  free(compressed_Ls);
  return timediff(start, stop);
}

unsigned long collect_lagrad(GMMInput gmm_input, double *ref_alphas,
                             double *ref_means, double *ref_icf,
                             double *temp_icf) {
  int d = gmm_input.d;
  int k = gmm_input.k;
  int n = gmm_input.n;
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  GMMGrad res = lagrad_main_term(
      deadbeef, gmm_input.alphas, 0, k, 1, deadbeef, gmm_input.means, 0, k, d,
      d, 1, deadbeef, gmm_input.Qs, 0, k, d, d, 1, deadbeef, gmm_input.Ls, 0, k,
      d, d, d * d, d, 1, deadbeef, gmm_input.x, 0, n, d, d, 1);
  gettimeofday(&stop, NULL);

  convert_ql_to_icf(d, k, n, res.dqs.aligned, res.dls.aligned, temp_icf);
  check_gmm_err(d, k, n, res.dalphas.aligned, ref_alphas, res.dmeans.aligned,
                ref_means, temp_icf, ref_icf, "LAGrad Full");
  free(res.dalphas.aligned);
  free(res.dmeans.aligned);
  free(res.dqs.aligned);
  free(res.dls.aligned);
  return timediff(start, stop);
}

unsigned long collect_enzyme_mlir_gradient(GMMInput gmm_input,
                                           double *ref_alphas,
                                           double *ref_means, double *ref_icf,
                                           double *temp_icf) {
  int d = gmm_input.d;
  int k = gmm_input.k;
  int n = gmm_input.n;
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
  GMMCompressedGrad res = enzyme_mlir_main_term_compressed(
      deadbeef, gmm_input.alphas, 0, k, 1, deadbeef, gmm_input.means, 0, k, d,
      d, 1, deadbeef, gmm_input.Qs, 0, k, d, d, 1, deadbeef, compressed_Ls, 0,
      k, tri_size, tri_size, 1, deadbeef, gmm_input.x, 0, n, d, d, 1);
  gettimeofday(&stop, NULL);
  convert_ql_compressed_to_icf(d, k, n, res.dqs.aligned, res.dls.aligned,
                               temp_icf);
  check_gmm_err(d, k, n, res.dalphas.aligned, ref_alphas, res.dmeans.aligned,
                ref_means, temp_icf, ref_icf, "Enzyme/MLIR Compressed");
  // print_d_arr_2d(res.dmeans.aligned, res.dmeans.size_0, res.dmeans.size_1);
  // print_d_arr_2d(res.dqs.aligned, res.dqs.size_0, res.dqs.size_1);
  // print_d_arr_2d(res.dls.aligned, res.dls.size_1, res.dls.size_2);
  free(res.dalphas.aligned);
  free(res.dmeans.aligned);
  free(res.dqs.aligned);
  free(res.dls.aligned);
  free(compressed_Ls);
  return timediff(start, stop);
}

void populate_ref(GMMInput gmm_input, double *ref_alphas, double *ref_means,
                  double *ref_icf) {
  int d = gmm_input.d;
  int k = gmm_input.k;
  int n = gmm_input.n;
  int tri_size = d * (d - 1) / 2;
  int icf_size = d * (d + 1) / 2;
  double *compressed_Ls = (double *)malloc(k * tri_size * sizeof(double));
  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < tri_size; j++) {
      compressed_Ls[i * tri_size + j] = gmm_input.icf[i * icf_size + d + j];
    }
  }
  GMMCompressedGrad res = enzyme_mlir_main_term_compressed(
      deadbeef, gmm_input.alphas, 0, k, 1, deadbeef, gmm_input.means, 0, k, d,
      d, 1, deadbeef, gmm_input.Qs, 0, k, d, d, 1, deadbeef, compressed_Ls, 0,
      k, tri_size, tri_size, 1, deadbeef, gmm_input.x, 0, n, d, d, 1);
  for (size_t i = 0; i < k; i++) {
    ref_alphas[i] = res.dalphas.aligned[i];
  }
  for (size_t i = 0; i < k * d; i++) {
    ref_means[i] = res.dmeans.aligned[i];
  }
  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < d; j++) {
      ref_icf[i * icf_size + j] = res.dqs.aligned[i * d + j];
    }
    for (size_t j = 0; j < tri_size; j++) {
      ref_icf[i * icf_size + j + d] = res.dls.aligned[i * tri_size + j];
    }
  }

  free(compressed_Ls);
}

int main() {
  GMMInput gmm_input = read_gmm_data("{{data_file}}");
  int d = gmm_input.d;
  int k = gmm_input.k;
  int icf_size = d * (d + 1) / 2;
  double *ref_alphas = (double *)malloc(k * sizeof(double));
  double *ref_means = (double *)malloc(d * k * sizeof(double));
  double *ref_icf = (double *)malloc(k * icf_size * sizeof(double));
  double *temp_icf = (double *)malloc(k * icf_size * sizeof(double));
  populate_ref(gmm_input, ref_alphas, ref_means, ref_icf);

  maintermBodyFunc funcs[] = {collect_handrolled_gradient,
                              collect_handrolled_compressed_gradient,
                              collect_lagrad, collect_enzyme_mlir_gradient};
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);
  for (size_t app = 0; app < num_apps; app++) {
    unsigned long results_df[NUM_RUNS];
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] =
          (*funcs[app])(gmm_input, ref_alphas, ref_means, ref_icf, temp_icf);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free(ref_alphas);
  free(ref_means);
  free(ref_icf);
  free(temp_icf);
  return 0;
}
