#include "mlir_c_abi.h"
#include "shared_types.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILENAME "benchmarks/data/gmm_d10_K25.txt"

typedef struct {
  double *data;
  double *aligned;
} UnrankedMemref;

extern UnrankedMemref gmm_objective(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

extern F64Descriptor3D lagrad_gmm(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

GMMInput read_data() {
  FILE *fp;
  GMMInput gmm_input;
  fp = fopen(FILENAME, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file \"%s\"\n", FILENAME);
    exit(EXIT_FAILURE);
  }

  int d, k, n;
  fscanf(fp, "%d %d %d", &d, &k, &n);

  // For some reason, this is consistent with Enzyme's implementation but the
  // original paper has d * (d - 1) / 2. When I change this value to "fix" it,
  // the gradient computation goes off slightly.
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
  gmm_input.x = x;
  gmm_input.wishart_gamma = wishart_gamma;
  gmm_input.wishart_m = wishart_m;
  return gmm_input;
}

void free_gmm_input(GMMInput gmm_input) {
  free(gmm_input.alphas);
  free(gmm_input.means);
  free(gmm_input.Qs);
  free(gmm_input.Ls);
  free(gmm_input.x);
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
double *collapse_ltri(F64Descriptor3D expanded) {
  int d = expanded.size_1;
  int icf_sz = expanded.size_1 * (expanded.size_1 - 1) / 2;
  double *result = (double *)malloc(expanded.size_0 * icf_sz * sizeof(double));
  for (size_t i = 0; i < expanded.size_0; i++) {
    size_t icf_idx = 0;
    for (size_t j = 0; j < d; j++) {
      for (size_t k = j + 1; k < d; k++) {
        result[i * icf_sz + icf_idx] = expanded.aligned[i * d * d + k * d + j];
        icf_idx++;
      }
    }
  }

  return result;
}

int main() {
  GMMInput gmm_input = read_data();
  printf("d %d k %d n %d wishart_gamma: %f\n", gmm_input.d, gmm_input.k,
         gmm_input.n, gmm_input.wishart_gamma);
  int d = gmm_input.d;
  int k = gmm_input.k;
  int n = gmm_input.n;

  double *deadbeef = (double *)0xdeadbeef;
  // gmm_objective(
  //     /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
  //     gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d,
  //     d, 1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
  //     /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
  //     /*wishart_gamma=*/gmm_input.wishart_gamma,
  //     /*wishart_m=*/gmm_input.wishart_m);
  F64Descriptor3D primal_result = lagrad_gmm(
      /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
      gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d, d,
      1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
      /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input.wishart_gamma,
      /*wishart_m=*/gmm_input.wishart_m);

  printf("adjoint for first arg result (%lld x %lld):\n", primal_result.size_0,
         primal_result.size_1);
  double *collapsed = collapse_ltri(primal_result);
  int icf_size = d * (d - 1) / 2;
  // It's easier just to look at a single collapsed matrix in the batch.
  print_d_arr(collapsed + icf_size, icf_size);
  free_gmm_input(gmm_input);
  free(collapsed);
}

// extern F32Descriptor0D gmm_objective(
//     /*alphas=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
//     /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
//     int64_t,
//     /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
//     /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
//     int64_t, int64_t,
//     /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
//     /*wishart_gamma=*/double,
//     /*wishart_m=*/int64_t);