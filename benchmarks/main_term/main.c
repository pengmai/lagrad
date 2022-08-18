#include "gmm.h"
#include "lagrad_utils.h"
#include <math.h>
#include <pmmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <xmmintrin.h>

#define NUM_RUNS 2
#define DISABLE_CHECKS 0

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
extern GMMCompressedGrad handwritten_main_term_compressed_buf_grad(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
// extern GMMGrad lagrad_main_term(
//     /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
//     /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
//     int64_t,
//     /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
//     /*Ls=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
//     int64_t, int64_t,
//     /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern GMMCompressedGrad lagrad_main_term(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern GMMCompressedGrad enzyme_mlir_main_term_compressed(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern void enzyme_c_main_term(int d, int k, int n, double *alphas,
                               double *alphasb, double *means, double *meansb,
                               double *Qs, double *Qsb, double *Ls, double *Lsb,
                               double *x);
extern void
manual_c_main_term(int d, int k, int n, double const *restrict alphas,
                   double *restrict alphasb, double const *restrict means,
                   double *restrict meansb, double const *restrict Qs,
                   double *restrict Qsb, double const *restrict Ls,
                   double *restrict Lsb, double const *restrict x);
extern void main_term_raised(int d, int k, int n, double const *restrict alphas,
                             double *restrict alphasb,
                             double const *restrict means,
                             double *restrict meansb, double const *restrict Qs,
                             double *restrict Qsb, double const *restrict Ls,
                             double *restrict Lsb, double const *restrict x);

typedef unsigned long (*maintermBodyFunc)(GMMInput gmm_input,
                                          double *ref_alphas, double *ref_means,
                                          double *ref_icf, double *temp_icf);

unsigned long collect_handrolled_tri_gradient(GMMInput gmm_input,
                                              double *ref_alphas,
                                              double *ref_means,
                                              double *ref_icf,
                                              double *temp_icf) {
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
  if (!DISABLE_CHECKS) {
    check_gmm_err(d, k, n, res.dalphas.aligned, ref_alphas, res.dmeans.aligned,
                  ref_means, temp_icf, ref_icf, "Handrolled full");
  }
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
  if (!DISABLE_CHECKS) {
    check_gmm_err(d, k, n, res.dalphas.aligned, ref_alphas, res.dmeans.aligned,
                  ref_means, temp_icf, ref_icf,
                  "Handwritten Compressed Tensor");
  }
  // print_d_arr_2d(res.dmeans.aligned, res.dmeans.size_0, res.dmeans.size_1);
  // print_d_arr_2d(res.dqs.aligned, res.dqs.size_0, res.dqs.size_1);
  // print_d_arr_2d(res.dls.aligned, res.dls.size_0, res.dls.size_1);
  free(res.dalphas.aligned);
  free(res.dmeans.aligned);
  free(res.dqs.aligned);
  free(res.dls.aligned);
  free(compressed_Ls);
  return timediff(start, stop);
}

// unsigned long collect_handrolled_compressed_bufferized_gradient(
//     GMMInput gmm_input, double *ref_alphas, double *ref_means, double
//     *ref_icf, double *temp_icf) {
//   int d = gmm_input.d;
//   int k = gmm_input.k;
//   int n = gmm_input.n;
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
//   GMMCompressedGrad res = handwritten_main_term_compressed_buf_grad(
//       deadbeef, gmm_input.alphas, 0, k, 1, deadbeef, gmm_input.means, 0, k,
//       d, d, 1, deadbeef, gmm_input.Qs, 0, k, d, d, 1, deadbeef,
//       compressed_Ls, 0, k, tri_size, tri_size, 1, deadbeef, gmm_input.x, 0,
//       n, d, d, 1);
//   gettimeofday(&stop, NULL);
//   convert_ql_compressed_to_icf(d, k, n, res.dqs.aligned, res.dls.aligned,
//                                temp_icf);
//   if (!DISABLE_CHECKS) {
//     check_gmm_err(d, k, n, res.dalphas.aligned, ref_alphas,
//     res.dmeans.aligned,
//                   ref_means, temp_icf, ref_icf,
//                   "Handwritten Compressed Bufferized");
//   }
//   // print_d_arr_2d(res.dmeans.aligned, res.dmeans.size_0,
//   res.dmeans.size_1);
//   // print_d_arr_2d(res.dqs.aligned, res.dqs.size_0, res.dqs.size_1);
//   // print_d_arr_2d(res.dls.aligned, res.dls.size_0, res.dls.size_1);
//   free(res.dalphas.aligned);
//   free(res.dmeans.aligned);
//   free(res.dqs.aligned);
//   free(res.dls.aligned);
//   free(compressed_Ls);
//   return timediff(start, stop);
// }

unsigned long collect_lagrad(GMMInput gmm_input, double *ref_alphas,
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
  gettimeofday(&start, 0);
  // GMMGrad res = lagrad_main_term(
  //     deadbeef, gmm_input.alphas, 0, k, 1, deadbeef, gmm_input.means, 0, k,
  //     d, d, 1, deadbeef, gmm_input.Qs, 0, k, d, d, 1, deadbeef, gmm_input.Ls,
  //     0, k, d, d, d * d, d, 1, deadbeef, gmm_input.x, 0, n, d, d, 1);
  GMMCompressedGrad res = lagrad_main_term(
      deadbeef, gmm_input.alphas, 0, k, 1, deadbeef, gmm_input.means, 0, k, d,
      d, 1, deadbeef, gmm_input.Qs, 0, k, d, d, 1, deadbeef, compressed_Ls, 0,
      k, tri_size, tri_size, 1, deadbeef, gmm_input.x, 0, n, d, d, 1);
  gettimeofday(&stop, 0);

  // convert_ql_to_icf(d, k, n, res.dqs.aligned, res.dls.aligned, temp_icf);
  convert_ql_compressed_to_icf(d, k, n, res.dqs.aligned, res.dls.aligned,
                               temp_icf);
  if (!DISABLE_CHECKS) {
    check_gmm_err(d, k, n, res.dalphas.aligned, ref_alphas, res.dmeans.aligned,
                  ref_means, temp_icf, ref_icf, "LAGrad Full");
  }
  free(res.dalphas.aligned);
  free(res.dmeans.aligned);
  free(res.dqs.aligned);
  free(res.dls.aligned);
  free(compressed_Ls);
  return timediff(start, stop);
}

unsigned long collect_enzyme_c_gradient(GMMInput gmm_input, double *ref_alphas,
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
  double *dalphas = calloc(k, sizeof(double));
  double *dmeans = calloc(k * d, sizeof(double));
  double *dQs = calloc(k * d, sizeof(double));
  double *dLs = calloc(k * tri_size, sizeof(double));
  enzyme_c_main_term(d, k, n, gmm_input.alphas, dalphas, gmm_input.means,
                     dmeans, gmm_input.Qs, dQs, compressed_Ls, dLs,
                     gmm_input.x);
  gettimeofday(&stop, NULL);
  convert_ql_compressed_to_icf(d, k, n, dQs, dLs, temp_icf);
  if (!DISABLE_CHECKS) {
    check_gmm_err(d, k, n, dalphas, ref_alphas, dmeans, ref_means, temp_icf,
                  ref_icf, "Enzyme/C Compressed");
  }
  // print_d_arr(dalphas, k);
  // print_d_arr_2d(dmeans, k, d);
  // print_d_arr_2d(dLs, k, tri_size);
  // print_d_arr_2d(dQs, k, d);
  free(compressed_Ls);
  free(dalphas);
  free(dmeans);
  free(dQs);
  free(dLs);
  return timediff(start, stop);
}

// unsigned long collect_raised_gradient(GMMInput gmm_input, double *ref_alphas,
//                                       double *ref_means, double *ref_icf,
//                                       double *temp_icf) {
//   int d = gmm_input.d;
//   int k = gmm_input.k;
//   int n = gmm_input.n;
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
//   double *dalphas = calloc(k, sizeof(double));
//   double *dmeans = calloc(k * d, sizeof(double));
//   double *dQs = calloc(k * d, sizeof(double));
//   double *dLs = calloc(k * tri_size, sizeof(double));
//   main_term_raised(d, k, n, gmm_input.alphas, dalphas, gmm_input.means,
//   dmeans,
//                    gmm_input.Qs, dQs, compressed_Ls, dLs, gmm_input.x);
//   gettimeofday(&stop, NULL);
//   convert_ql_compressed_to_icf(d, k, n, dQs, dLs, temp_icf);
//   if (!DISABLE_CHECKS) {
//     check_gmm_err(d, k, n, dalphas, ref_alphas, dmeans, ref_means, temp_icf,
//                   ref_icf, "Manual Raised Compressed");
//   }
//   // print_d_arr(dalphas, k);
//   // print_d_arr_2d(dmeans, k, d);
//   // print_d_arr_2d(dQs, k, d);
//   // print_d_arr_2d(dLs, k, tri_size);
//   free(compressed_Ls);
//   free(dalphas);
//   free(dmeans);
//   free(dQs);
//   free(dLs);
//   return timediff(start, stop);
// }

unsigned long collect_manual_c_gradient(GMMInput gmm_input, double *ref_alphas,
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
  clock_t begin, end;
  gettimeofday(&start, NULL);
  begin = clock();
  double *dalphas = calloc(k, sizeof(double));
  double *dmeans = calloc(k * d, sizeof(double));
  double *dQs = calloc(k * d, sizeof(double));
  double *dLs = calloc(k * tri_size, sizeof(double));
  manual_c_main_term(d, k, n, gmm_input.alphas, dalphas, gmm_input.means,
                     dmeans, gmm_input.Qs, dQs, compressed_Ls, dLs,
                     gmm_input.x);
  end = clock();
  gettimeofday(&stop, NULL);
  convert_ql_compressed_to_icf(d, k, n, dQs, dLs, temp_icf);
  if (!DISABLE_CHECKS) {
    check_gmm_err(d, k, n, dalphas, ref_alphas, dmeans, ref_means, temp_icf,
                  ref_icf, "C Manual");
  }
  // print_d_arr(dalphas, k);
  // print_d_arr_2d(res.dmeans.aligned, res.dmeans.size_0, res.dmeans.size_1);
  // print_d_arr_2d(res.dqs.aligned, res.dqs.size_0, res.dqs.size_1);
  // print_d_arr_2d(res.dls.aligned, res.dls.size_1, res.dls.size_2);
  free(compressed_Ls);
  free(dalphas);
  free(dmeans);
  free(dQs);
  free(dLs);
  return timediff(start, stop);
  // return end - begin;
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
  if (!DISABLE_CHECKS) {
    check_gmm_err(d, k, n, res.dalphas.aligned, ref_alphas, res.dmeans.aligned,
                  ref_means, temp_icf, ref_icf, "Enzyme/MLIR Compressed");
  }
  // print_d_arr(res.dalphas.aligned, res.dalphas.size);
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

typedef struct MainTermApp {
  const char *name;
  maintermBodyFunc func;
} MainTermApp;

int main() {
  // _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  // _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  GMMInput gmm_input = read_gmm_data("{{data_file}}");
  int d = gmm_input.d;
  int k = gmm_input.k;
  int icf_size = d * (d + 1) / 2;
  double *ref_alphas = (double *)malloc(k * sizeof(double));
  double *ref_means = (double *)malloc(d * k * sizeof(double));
  double *ref_icf = (double *)malloc(k * icf_size * sizeof(double));
  double *temp_icf = (double *)malloc(k * icf_size * sizeof(double));
  if (!DISABLE_CHECKS) {
    populate_ref(gmm_input, ref_alphas, ref_means, ref_icf);
  }
  MainTermApp apps[] = {
      // Need a comment here for compatibility with jinja
      {.name = "Enzyme/C", .func = collect_enzyme_c_gradient},
      // {.name = "Raised", .func = collect_raised_gradient},
      {.name = "LAGrad Comp", .func = collect_lagrad},
      // {.name = "Manual Tri", .func = collect_handrolled_tri_gradient},
      {.name = "Manual", .func = collect_manual_c_gradient},
      {.name = "Manual/MLIR/Tensor",
       .func = collect_handrolled_compressed_gradient},
      // {.name = "Manual/MLIR/Bufferized",
      //  .func = collect_handrolled_compressed_bufferized_gradient}
  };

  // maintermBodyFunc funcs[] = {
  //     // collect_handrolled_gradient,
  //     // collect_handrolled_compressed_gradient,
  //     collect_lagrad, collect_enzyme_c_gradient, collect_raised_gradient,
  //     collect_manual_c_gradient,
  //     // collect_enzyme_mlir_gradient,
  // };
  size_t num_apps = sizeof(apps) / sizeof(apps[0]);
  unsigned long results_df[NUM_RUNS];
  for (size_t app = 0; app < num_apps; app++) {
    printf("%s: ", apps[app].name);
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] =
          apps[app].func(gmm_input, ref_alphas, ref_means, ref_icf, temp_icf);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free(ref_alphas);
  free(ref_means);
  free(ref_icf);
  free(temp_icf);
  return 0;
}
