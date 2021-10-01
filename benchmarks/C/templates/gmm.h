#pragma once
#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define GRAD_FILENAME "benchmarks/data/gmm_results.txt"

void read_gmm_grads(size_t d, size_t k, size_t n, double *dalphas,
                    double *dmeans, double *dicf) {
  FILE *fp = fopen(GRAD_FILENAME, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file \"%s\"\n", GRAD_FILENAME);
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

void serialize_gmm_grads(size_t d, size_t k, size_t n, double *dalphas,
                         double *dmeans, double *dicf) {
  FILE *fp;
  fp = fopen(GRAD_FILENAME, "w");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file \"%s\"\n", GRAD_FILENAME);
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
  for (size_t i = 0; i < k; i++) {
    alphas_err += fabs(ref_alphas[i] - dalphas[i]);
  }
  // Tolerance here is relatively high because of precision lost when
  // serializing the floats.
  if (alphas_err > 1e-4) {
    printf("(%s) alphas err: %f\n", app, alphas_err);
  }

  double means_err = 0.0;
  for (size_t i = 0; i < k * d; i++) {
    means_err += fabs(ref_means[i] - dmeans[i]);
  }
  if (means_err > 1e-4) {
    printf("(%s) means err: %f\n", app, means_err);
  }

  double icf_err = 0.0;
  int icf_sz = d * (d + 1) / 2;
  for (size_t i = 0; i < k * icf_sz; i++) {
    if (i < 10) {
      printf("expected: %f actual: %f\n", ref_icf[i], dicf[i]);
    }
    icf_err += fabs(ref_icf[i] - dicf[i]);
  }
  if (icf_err > 1e-4) {
    printf("(%s) icf err: %f\n", app, icf_err);
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
void collapse_ltri(size_t k, size_t d, double *Ls, double *out, size_t outer) {
  // icf_size = 55, d = 10, k = 25
  int ltri_size = d * (d - 1) / 2;
  for (size_t i = 0; i < k; i++) {
    size_t icf_idx = 0;
    for (size_t j = 0; j < d; j++) {
      for (size_t l = j + 1; l < d; l++) {
        // out[i * ltri_size + icf_idx] = Ls[(i * d * d) + (l * d) + j];
        out[i * ltri_size + icf_idx] = Ls[(l * d) + j];
        if (outer == k - 1 && i == 0) {
          printf("(collapse ltri) out index: %lu\n", i * ltri_size + icf_idx);
        }
        icf_idx++;
      }
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
    collapse_ltri(k, d, &Ls[d0 * d * d], &icf[d0 * icf_sz + d], d0);
  }
  print_d_arr(icf + icf_sz, icf_sz);
}
