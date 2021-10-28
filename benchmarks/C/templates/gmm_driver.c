#include "gmm.h"
#include "mlir_c_abi.h"
#include "shared_types.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define NUM_RUNS 20
#define NUM_WARMUPS 20

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
extern void dgmm_objective_full_L(GMMInput *gmm, double *alphasb,
                                  double *meansb, double *Qsb, double *Lsb,
                                  double *err, double *errb);

// Running the forward pass to see if that segfaults
extern double enzyme_gmm_primal(GMMInput gmm);

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

int main() {
  GMMInput gmm_input = read_gmm_data();
  int d = gmm_input.d;
  int k = gmm_input.k;
  int n = gmm_input.n;
  int icf_size = d * (d + 1) / 2;
  double *deadbeef = (double *)0xdeadbeef;

  unsigned long grad_results[NUM_RUNS + NUM_WARMUPS];
  unsigned long enzyme_results[NUM_RUNS + NUM_WARMUPS];
  unsigned long enzyme_notri_results[NUM_RUNS + NUM_WARMUPS];

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

  for (size_t run = 0; run < NUM_RUNS + NUM_WARMUPS; run++) {
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

    enzyme_results[run] = timediff(start, stop);
    check_gmm_err(d, k, n, alphasb, ref_alphas, meansb, ref_means, icfb,
                  ref_icf, "Enzyme");

    free(alphasb);
    free(meansb);
    free(icfb);
  }

  for (size_t run = 0; run < NUM_RUNS + NUM_WARMUPS; run++) {
    struct timeval start, stop;

    gettimeofday(&start, NULL);
    GMMGrad lagrad_result = lagrad_gmm(
        /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
        gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d,
        d, 1, /*Ls=*/deadbeef, DELETEME_Ls, 0, k, d * (d - 1) / 2,
        d * (d - 1) / 2, 1,
        /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
        /*wishart_gamma=*/gmm_input.wishart_gamma,
        /*wishart_m=*/gmm_input.wishart_m);
    gettimeofday(&stop, NULL);

    grad_results[run] = timediff(start, stop);
    // convert_ql_to_icf(d, k, n, lagrad_result.dqs.aligned,
    //                   lagrad_result.dls.aligned, temp_icf);
    // check_gmm_err(d, k, n, lagrad_result.dalphas.aligned, ref_alphas,
    //               lagrad_result.dmeans.aligned, ref_means, temp_icf, ref_icf,
    //               "LAGrad");
    // serialize_gmm_grads(d, k, n, lagrad_result.dalphas.aligned,
    //                     lagrad_result.dmeans.aligned, icfb);

    free(lagrad_result.dalphas.aligned);
    free(lagrad_result.dmeans.aligned);
    free(lagrad_result.dqs.aligned);
    free(lagrad_result.dls.aligned);
  }

  for (size_t run = 0; run < NUM_RUNS + NUM_WARMUPS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    double *alphasb = (double *)malloc(k * sizeof(double));
    double *meansb = (double *)malloc(d * k * sizeof(double));
    double *Qsb = (double *)malloc(d * k * sizeof(double));
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
    enzyme_notri_results[run] = timediff(start, stop);

    // convert_ql_to_icf(d, k, n, Qsb, Lsb, temp_icf);
    // check_gmm_err(d, k, n, alphasb, ref_alphas, meansb, ref_means, temp_icf,
    //               ref_icf, "Enzyme full");
    free(alphasb);
    free(meansb);
    free(Qsb);
    free(Lsb);
  }

  double lagrad_avg = 0.0;
  double enzyme_avg = 0.0;
  double enzyme_notri_avg = 0.0;
  for (size_t run = NUM_WARMUPS; run < NUM_WARMUPS + NUM_RUNS; run++) {
    lagrad_avg += grad_results[run];
    enzyme_notri_avg += enzyme_notri_results[run];
    enzyme_avg += enzyme_results[run];
  }
  printf("LAGrad avg: %f\n", lagrad_avg / NUM_RUNS);
  printf("Enzyme avg: %f\n", enzyme_avg / NUM_RUNS);
  printf("Enzyme avg (notri): %f\n", enzyme_notri_avg / NUM_RUNS);
  printf("Slowdown: %f\n", lagrad_avg / enzyme_avg);

  free_gmm_input(gmm_input);
  free(ref_alphas);
  free(ref_means);
  free(ref_icf);
}
