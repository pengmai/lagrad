#include "gmm.h"
#include "mlir_c_abi.h"
#include "shared_types.h"
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

extern GMMGrad lagrad_gmm_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

extern GMMGrad enzyme_gmm_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

typedef unsigned long (*bodyFunc)(GMMInput gmm_input, double *ref_alphas,
                                  double *ref_means, double *ref_icf,
                                  double *temp_icf);

unsigned long collect_lagrad_full(GMMInput gmm_input, double *ref_alphas,
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

  convert_ql_to_icf(d, k, n, ans.dqs.aligned, ans.dls.aligned, temp_icf);
  check_gmm_err(d, k, n, ans.dalphas.aligned, ref_alphas, ans.dmeans.aligned,
                ref_means, temp_icf, ref_icf, "LAGrad (ADBench) Full");
  free(ans.dalphas.aligned);
  free(ans.dmeans.aligned);
  free(ans.dqs.aligned);
  free(ans.dls.aligned);
  return timediff(start, stop);
}

unsigned long collect_enzyme_full(GMMInput gmm_input, double *ref_alphas,
                                  double *ref_means, double *ref_icf,
                                  double *temp_icf) {
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  GMMGrad ans = enzyme_gmm_full(
      /*alphas=*/deadbeef, gmm_input.alphas, 0, k, 1, /*means=*/deadbeef,
      gmm_input.means, 0, k, d, d, 1, /*Qs=*/deadbeef, gmm_input.Qs, 0, k, d, d,
      1, /*Ls=*/deadbeef, gmm_input.Ls, 0, k, d, d, d * d, d, 1,
      /*x=*/deadbeef, gmm_input.x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input.wishart_gamma,
      /*wishart_m=*/gmm_input.wishart_m);
  gettimeofday(&stop, NULL);

  convert_ql_to_icf(d, k, n, ans.dqs.aligned, ans.dls.aligned, temp_icf);
  check_gmm_err(d, k, n, ans.dalphas.aligned, ref_alphas, ans.dmeans.aligned,
                ref_means, temp_icf, ref_icf, "Enzyme (ADBench) Full");
  free(ans.dalphas.aligned);
  free(ans.dmeans.aligned);
  free(ans.dqs.aligned);
  free(ans.dls.aligned);
  return timediff(start, stop);
}

int main() {
  GMMInput gmm_input = read_gmm_data();
  int d = gmm_input.d;
  int k = gmm_input.k;
  int n = gmm_input.n;
  int icf_size = d * (d + 1) / 2;
  // assert(d == {{d}});

  unsigned long results_df[NUM_RUNS];
  double *ref_alphas = (double *)malloc(k * sizeof(double));
  double *ref_means = (double *)malloc(d * k * sizeof(double));
  double *ref_icf = (double *)malloc(k * icf_size * sizeof(double));
  double *temp_icf = (double *)malloc(k * icf_size * sizeof(double));

  read_gmm_grads(d, k, n, ref_alphas, ref_means, ref_icf);

  bodyFunc funcs[] = {collect_lagrad_full};
  // collect_enzyme_full};
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);

  for (size_t app = 0; app < num_apps; app++) {
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] =
          funcs[app](gmm_input, ref_alphas, ref_means, ref_icf, temp_icf);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  printf("Done\n");

  free(ref_alphas);
  free(ref_means);
  free(ref_icf);
  free(temp_icf);
}
