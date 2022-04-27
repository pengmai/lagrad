#pragma once
#include "mlir_c_abi.h"
#include "shared_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// #define FILENAME "benchmarks/data/gmm_d10_K25.txt"
// #define GRAD_FILENAME "benchmarks/data/gmm_results.txt"
// #define FILENAME "benchmarks/data/gmm_d128_K200.txt"
// #define GRAD_FILENAME "benchmarks/data/gmm_d128_K200_results.txt"

GMMInput read_gmm_data(const char*data_file) {
  FILE *fp;
  GMMInput gmm_input;
  fp = fopen(data_file, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file \"%s\"\n", data_file);
    exit(EXIT_FAILURE);
  }

  int d, k, n;
  fscanf(fp, "%d %d %d", &d, &k, &n);

  int icf_sz = d * (d + 1) / 2;
  double *alphas = (double *)malloc(k * sizeof(double));
  double *means = (double *)malloc(d * k * sizeof(double));
  double *icf = (double *)malloc(icf_sz * k * sizeof(double));
  double *Qs = (double *)malloc(k * d * sizeof(double));
  double *Ls = (double *)malloc(k * d * d * sizeof(double));
  for (int i = 0; i < k * d * d; i++) {
    Ls[i] = 0;
  }

  double *x = (double *)malloc(d * n * sizeof(double));

  for (int i = 0; i < k; i++) {
    fscanf(fp, "%lf", &alphas[i]);
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < d; j++) {
      fscanf(fp, "%lf", &means[i * d + j]);
    }
  }

  for (int i = 0; i < k; i++) {
    int Lcol = 0;
    int Lidx = 0;
    for (int j = 0; j < icf_sz; j++) {
      fscanf(fp, "%lf", &icf[i * icf_sz + j]);
      if (j < d) {
        Qs[i * d + j] = icf[i * icf_sz + j];
      } else {
        Ls[i * d * d + (Lidx + 1) * d + Lcol] = icf[i * icf_sz + j];
        Lidx++;
        if (Lidx == d - 1) {
          Lcol++;
          Lidx = Lcol;
        }
      }
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      fscanf(fp, "%lf", &x[i * d + j]);
    }
  }

  int wishart_m;
  double wishart_gamma;
  fscanf(fp, "%lf %d", &wishart_gamma, &wishart_m);
  fclose(fp);

  gmm_input.d = d;
  gmm_input.k = k;
  gmm_input.n = n;
  gmm_input.alphas = alphas;
  gmm_input.means = means;
  gmm_input.Qs = Qs;
  gmm_input.Ls = Ls;
  gmm_input.icf = icf;
  gmm_input.x = x;
  gmm_input.wishart_gamma = wishart_gamma;
  gmm_input.wishart_m = wishart_m;
  return gmm_input;
}

void read_gmm_grads(const char *grad_file, size_t d, size_t k, size_t n, double *dalphas,
                    double *dmeans, double *dicf) {
  FILE *fp = fopen(grad_file, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file \"%s\"\n", grad_file);
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < k; i++) {
    fscanf(fp, "%lf", &dalphas[i]);
  }

  for (size_t i = 0; i < k * d; i++) {
    fscanf(fp, "%lf", &dmeans[i]);
  }

  int icf_sz = d * (d + 1) / 2;
  for (size_t i = 0; i < k * icf_sz; i++) {
    fscanf(fp, "%lf", &dicf[i]);
  }

  fclose(fp);
}

void serialize_gmm_grads(const char* grad_file, size_t d, size_t k, size_t n, double *dalphas,
                         double *dmeans, double *dicf) {
  FILE *fp;
  fp = fopen(grad_file, "w");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file \"%s\"\n", grad_file);
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < k; i++) {
    fprintf(fp, "%lf", dalphas[i]);
    if (i != k - 1) {
      fprintf(fp, " ");
    }
  }
  fprintf(fp, "\n");

  for (size_t i = 0; i < d * k; i++) {
    fprintf(fp, "%lf", dmeans[i]);
    if (i != d * k - 1) {
      fprintf(fp, " ");
    }
  }
  fprintf(fp, "\n");

  int icf_sz = d * (d + 1) / 2;
  for (size_t i = 0; i < k * icf_sz; i++) {
    fprintf(fp, "%lf", dicf[i]);
    if (i != k * icf_sz - 1) {
      fprintf(fp, " ");
    }
  }
  fprintf(fp, "\n");

  fclose(fp);
}

void check_gmm_err(size_t d, size_t k, size_t n, double *dalphas,
                   double *ref_alphas, double *dmeans, double *ref_means,
                   double *dicf, double *ref_icf, const char *app) {
  double alphas_err = 0.0;
  int hasNaN = 0;
  for (size_t i = 0; i < k; i++) {
    hasNaN |= isnan(dalphas[i]);
    alphas_err += fabs(ref_alphas[i] - dalphas[i]);
  }
  if (hasNaN) {
    printf("(%s) alphas has NaN values\n", app);
  }
  // Tolerance here is relatively high because of precision lost when
  // serializing the floats.
  if (alphas_err > 1e-4) {
    printf("(%s) alphas err: %f\n", app, alphas_err);
  }

  double max_means_err = -1;
  hasNaN = 0;
  for (size_t i = 0; i < k * d; i++) {
    hasNaN |= isnan(dmeans[i]);
    if (fabs(ref_means[i] - dmeans[i]) > max_means_err) {
      max_means_err = fabs(ref_means[i] - dmeans[i]);
    }
  }
  if (hasNaN) {
    printf("(%s) means has NaN values\n", app);
  }
  if (max_means_err > 1e-6) {
    printf("(%s) max means err: %f\n", app, max_means_err);
  }

  int icf_sz = d * (d + 1) / 2;
  double max_icf_err = -1;
  hasNaN = 0;
  for (size_t i = 0; i < k * icf_sz; i++) {
    hasNaN |= isnan(dicf[i]);
    if (fabs(ref_icf[i] - dicf[i]) > max_icf_err) {
      max_icf_err = fabs(ref_icf[i] - dicf[i]);
    }
  }
  if (hasNaN) {
    printf("(%s) icf has NaN values\n", app);
  }
  if (max_icf_err > 1e-5) {
    printf("(%s) max icf err: %f\n", app, max_icf_err);
  }
}

/**
 * Convert a batch of lower triangular matrix into its packed representation.
 * Example:
 * [[0 0 0 0]
 *  [a 0 0 0]
 *  [b d 0 0]
 *  [c e f 0]]
 * becomes:
 * [a b c d e f]
 */
void collapse_ltri(size_t d, double *Ls, double *out) {
  size_t icf_idx = 0;
  for (size_t j = 0; j < d; j++) {
    for (size_t l = j + 1; l < d; l++) {
      out[icf_idx] = Ls[(l * d) + j];
      icf_idx++;
    }
  }
}

/**
 * Converts the split Q and L matrices to icf
 */
void convert_ql_to_icf(size_t d, size_t k, size_t n, double *Qs, double *Ls,
                       double *icf) {
  int icf_sz = d * (d + 1) / 2;
  for (size_t d0 = 0; d0 < k; d0++) {
    for (size_t d1 = 0; d1 < d; d1++) {
      icf[d0 * icf_sz + d1] = Qs[d0 * d + d1];
    }
    collapse_ltri(d, &Ls[d0 * d * d], &icf[d0 * icf_sz + d]);
  }
}

void convert_ql_compressed_to_icf(size_t d, size_t k, size_t n, double *Qs,
                                  double *Ls_compressed, double *icf) {
  int icf_sz = d * (d + 1) / 2;
  int tri_sz = d * (d - 1) / 2;
  double *temp_L = (double *)malloc(k * d * d * sizeof(double));
  for (size_t d0 = 0; d0 < k; d0++) {
    for (size_t d1 = 0; d1 < d; d1++) {
      icf[d0 * icf_sz + d1] = Qs[d0 * d + d1];
    }

    for (size_t d1 = 0; d1 < tri_sz; d1++) {
      icf[d0 * icf_sz + d + d1] = Ls_compressed[d0 * tri_sz + d1];
    }

    // size_t idx = 0;
    // for (size_t d1 = 0; d1 < d; d1++) {
    //   for (size_t d2 = 0; d2 < d1; d2++) {
    //     temp_L[d0 * d * d + d1 * d + d2] = Ls_compressed[idx];
    //     idx++;
    //   }
    // }

    // collapse_ltri(d, &temp_L[d0 * d * d], &icf[d0 * icf_sz + d]);
  }
  free(temp_L);
}
