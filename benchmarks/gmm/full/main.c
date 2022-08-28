#include "gmm.h"
#include "gmm_types.h"
#include "lagrad_utils.h"
#include "memusage.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define NUM_RUNS 1
#define CHECK_MEM 0

RunProcDyn rpd;
void check_mem_usage() {
  run_get_dynamic_proc_info(getpid(), &rpd);
  printf("%zu\t%zu\n", rpd.rss, rpd.vsize);
}

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

extern double emgmm_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);
extern GMMGrad enzyme_mlir_gmm_full(
    /*alphas=*/double *, double *, int64_t, int64_t, int64_t,
    /*means=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Qs=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*Ls*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*wishart_gamma=*/double,
    /*wishart_m=*/int64_t);

GMMGrad enzyme_mlir_gmm_full_adjoint(GMMInput *gmm_input) {
  int n = gmm_input->n, k = gmm_input->k, d = gmm_input->d;

  double debug = emgmm_full(
      /*alphas=*/deadbeef, gmm_input->alphas, 0, k, 1,
      /*means=*/deadbeef, gmm_input->means, 0, k, d, d, 1,
      /*Qs=*/deadbeef, gmm_input->Qs, 0, k, d, d, 1,
      /*Ls=*/deadbeef, gmm_input->Ls, 0, k, d, d, d * d, d, 1,
      /*x=*/deadbeef, gmm_input->x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input->wishart_gamma,
      /*wishart_m=*/gmm_input->wishart_m);
  printf("Debug: %f\n", debug);
  return enzyme_mlir_gmm_full(
      /*alphas=*/deadbeef, gmm_input->alphas, 0, k, 1,
      /*means=*/deadbeef, gmm_input->means, 0, k, d, d, 1,
      /*Qs=*/deadbeef, gmm_input->Qs, 0, k, d, d, 1,
      /*Ls=*/deadbeef, gmm_input->Ls, 0, k, d, d, d * d, d, 1,
      /*x=*/deadbeef, gmm_input->x, 0, n, d, d, 1,
      /*wishart_gamma=*/gmm_input->wishart_gamma,
      /*wishart_m=*/gmm_input->wishart_m);
}

void free_gmm_input(GMMInput gmm_input) {
  free(gmm_input.alphas);
  free(gmm_input.means);
  free(gmm_input.Qs);
  free(gmm_input.Ls);
  free(gmm_input.x);
}

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
  if (CHECK_MEM) {
    check_mem_usage();
  } else {
    convert_ql_to_icf(d, k, n, ans.dqs.aligned, ans.dls.aligned, temp_icf);
    check_gmm_err(d, k, n, ans.dalphas.aligned, ref_alphas, ans.dmeans.aligned,
                  ref_means, temp_icf, ref_icf, app.name);
  }
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
      // {.name = "Enzyme/C", .func = enzyme_c_gmm_full},
      {.name = "Enzyme/MLIR", .func = enzyme_mlir_gmm_full_adjoint},
  };

  size_t num_apps = sizeof(apps) / sizeof(apps[0]);

  unsigned long results_df[NUM_RUNS];
  double *ref_icf = calloc(k * icf_size, sizeof(double));
  double *temp_icf = calloc(k * icf_size, sizeof(double));
  GMMGrad ref_grad;
  if (!CHECK_MEM) {
    ref_grad = populate_ref(&gmm_input);
    convert_ql_to_icf(d, k, n, ref_grad.dqs.aligned, ref_grad.dls.aligned,
                      ref_icf);
    free(ref_grad.dqs.aligned);
    free(ref_grad.dls.aligned);
  }

  for (size_t app = 0; app < num_apps; app++) {
    printf("%s: ", apps[app].name);
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] =
          collect_full_adjoint(apps[app], &gmm_input, ref_grad.dalphas.aligned,
                               ref_grad.dmeans.aligned, ref_icf, temp_icf);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  if (!CHECK_MEM) {
    free(ref_grad.dalphas.aligned);
    free(ref_grad.dmeans.aligned);
  }
  free_gmm_input(gmm_input);
  free(ref_icf);
  free(temp_icf);
}
