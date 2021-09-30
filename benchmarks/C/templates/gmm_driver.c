#include "mlir_c_abi.h"
#include "shared_types.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define FILENAME "benchmarks/data/gmm_d10_K25.txt"
#define NUM_RUNS 1
#define NUM_WARMUPS 0

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

extern GMMGrad lagrad_gmm(
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

// Running the forward pass to see if that segfaults
extern double enzyme_gmm_primal(GMMInput gmm);

extern void enzyme_gmm_objective_full_L(int d, int k, int n, double *alphas,
                                        double *means, double *Qs, double *Ls,
                                        double *x, double wishart_gamma,
                                        int wishart_m, double *err);

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
  gmm_input.icf = icf;
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
  int d = gmm_input.d;
  int k = gmm_input.k;
  int n = gmm_input.n;
  int icf_size = d * (d - 1) / 2;
  double *deadbeef = (double *)0xdeadbeef;

  unsigned long *grad_results =
      (unsigned long *)malloc((NUM_RUNS + NUM_WARMUPS) * sizeof(unsigned long));
  unsigned long *enzyme_results =
      (unsigned long *)malloc((NUM_RUNS + NUM_WARMUPS) * sizeof(unsigned long));

  double *alphasb = (double *)malloc(k * sizeof(double));
  double *meansb = (double *)malloc(d * k * sizeof(double));
  double *icfb = (double *)malloc(icf_size * k * sizeof(double));

  for (size_t run = 0; run < NUM_RUNS + NUM_WARMUPS; run++) {
    struct timeval start, stop;

    gettimeofday(&start, NULL);
    GMMGrad lagrad_result = lagrad_gmm(
        /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
        gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d,
        d, 1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
        /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
        /*wishart_gamma=*/gmm_input.wishart_gamma,
        /*wishart_m=*/gmm_input.wishart_m);
    gettimeofday(&stop, NULL);

    grad_results[run] = timediff(start, stop);

    gettimeofday(&start, NULL);
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

    enzyme_results[run] = timediff(start, stop);

    double alphas_err = 0.0;
    // if (run == 0) {
    //   printf("Enzyme alphas: ");
    // }
    for (size_t i = 0; i < k; i++) {
      // printf("");
      alphas_err += fabs(alphasb[i] - lagrad_result.dalphas.aligned[i]);
    }
    if (alphas_err > 1e-6) {
      printf("(Run %lu) Alphas err: %f\n", run, alphas_err);
    }
    double means_err = 0.0;
    for (size_t i = 0; i < d * k; i++) {
      means_err += fabs(meansb[i] - lagrad_result.dmeans.aligned[i]);
    }
    if (means_err > 1e-6) {
      printf("(Run %lu) Means err: %f\n", run, means_err);
    }

    free(lagrad_result.dalphas.aligned);
    free(lagrad_result.dmeans.aligned);
    free(lagrad_result.dqs.aligned);
    free(lagrad_result.dls.aligned);
  }

  double lagrad_avg = 0.0;
  double enzyme_avg = 0.0;
  for (size_t run = NUM_WARMUPS; run < NUM_WARMUPS + NUM_RUNS; run++) {
    lagrad_avg += grad_results[run];
    enzyme_avg += enzyme_results[run];
  }
  printf("LAGrad avg: %f\n", lagrad_avg / NUM_RUNS);
  printf("Enzyme avg: %f\n", enzyme_avg / NUM_RUNS);
  printf("Slowdown: %f\n", lagrad_avg / enzyme_avg);

  double enzyme_primal_err = 0.0;
  enzyme_gmm_objective_full_L(d, k, n, gmm_input.alphas, gmm_input.means,
                              gmm_input.Qs, gmm_input.Ls, gmm_input.x,
                              gmm_input.wishart_gamma, gmm_input.wishart_m,
                              &enzyme_primal_err);
  double reference_err = enzyme_gmm_primal(gmm_input);
  printf("Primal err with full L: %f\n", enzyme_primal_err);
  printf("Primal err without full L: %f\n", reference_err);

  // printf("adjoint for icf: (%lld x %lld x %lld):\n", primal_result.size_0,
  //        primal_result.size_1, primal_result.size_2);
  // double *collapsed = collapse_ltri(primal_result);
  // It's easier just to look at a single collapsed matrix in the batch.
  // print_d_arr(collapsed, icf_size);

  // free(alphasb);
  // free(meansb);
  // free(icfb);
  // free(enzyme_results);
  // free(grad_results);
  // free_gmm_input(gmm_input);
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