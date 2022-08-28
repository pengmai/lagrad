#include "ba.h"
#include "lagrad_utils.h"

#define NUM_RUNS 6
#define CHECK_MEM 0

double *deadbeef = (double *)0xdeadbeef;

BAGrad enzyme_c_compute_reproj_err(double *cams, double *X, double *w,
                                   double *feats, double *g) {
  double *camsb_buf = calloc(BA_NCAMPARAMS, sizeof(double));
  double *Xb_buf = calloc(3, sizeof(double));
  F64Descriptor1D camsb = {.allocated = NULL,
                           .aligned = camsb_buf,
                           .offset = 0,
                           .size = BA_NCAMPARAMS,
                           .stride = 1};
  F64Descriptor1D Xb = {.allocated = NULL,
                        .aligned = Xb_buf,
                        .offset = 0,
                        .size = 3,
                        .stride = 1};
  double wb = 0.0;
  double err[2];
  dcompute_reproj_error(cams, camsb.aligned, X, Xb.aligned, w, &wb, feats, err,
                        g);
  BAGrad grad = {.dcam = camsb, .dX = Xb, .dw = wb};
  return grad;
}

double enzyme_c_compute_w_err(double w) {
  double wb = 0, err = 0, errb = 1.0;
  dcompute_w_error(&w, &wb, &err, &errb);
  return wb;
}

BAGrad lagrad_compute_reproj_error_wrapper(double *cams, double *X, double *w,
                                           double *feats, double *g) {
  return lagrad_compute_reproj_error(
      /*cams=*/deadbeef, cams, 0, BA_NCAMPARAMS, 1,
      /*X=*/deadbeef, X, 0, 3, 1, /*w=*/*w, deadbeef, feats, 0, 2, 1,
      /*g=*/deadbeef, g, 0, 2, 1);
}

BAGrad enzyme_compute_reproj_error_wrapper(double *cams, double *X, double *w,
                                           double *feats, double *g) {
  return enzyme_compute_reproj_error(
      /*cams=*/deadbeef, cams, 0, BA_NCAMPARAMS, 1,
      /*X=*/deadbeef, X, 0, 3, 1, /*w=*/*w, deadbeef, feats, 0, 2, 1,
      /*g=*/deadbeef, g, 0, 2, 1);
}

typedef struct BAApp {
  const char *name;
  BAGrad (*reproj_func)(double *cams, double *X, double *w, double *feats,
                        double *g);
  double (*w_func)(double w);
} BAApp;

void calculate_reproj_jacobian(BAApp app, BAInput ba_input, BASparseMat *J) {
  double errb[2];
  double reproj_err_d[2 * (BA_NCAMPARAMS + 3 + 1)];
  for (size_t i = 0; i < ba_input.p; i++) {
    int camIdx = ba_input.obs[2 * i + 0];
    int ptIdx = ba_input.obs[2 * i + 1];

    // Calculate first row
    errb[0] = 1.0;
    errb[1] = 0.0;
    BAGrad ans = app.reproj_func(&ba_input.cams[camIdx * BA_NCAMPARAMS],
                                 &ba_input.X[ptIdx * 3], &ba_input.w[i],
                                 &ba_input.feats[i * 2], errb);

    size_t j = 0;
    for (j = 0; j < BA_NCAMPARAMS; j++) {
      reproj_err_d[j * 2] = ans.dcam.aligned[j];
    }
    for (size_t k = 0; k < 3; k++) {
      reproj_err_d[j * 2] = ans.dX.aligned[k];
      j++;
    }
    reproj_err_d[j * 2] = ans.dw;
    free(ans.dcam.aligned);
    free(ans.dX.aligned);

    errb[1] = 0.0;
    errb[0] = 1.0;

    ans = app.reproj_func(&ba_input.cams[camIdx * BA_NCAMPARAMS],
                          &ba_input.X[ptIdx * 3], &ba_input.w[i],
                          &ba_input.feats[i * 2], errb);
    j = 0;
    for (j = 0; j < BA_NCAMPARAMS; j++) {
      reproj_err_d[j * 2 + 1] = ans.dcam.aligned[j];
    }
    for (size_t k = 0; k < 3; k++) {
      reproj_err_d[j * 2 + 1] = ans.dX.aligned[k];
      j++;
    }
    reproj_err_d[j * 2 + 1] = ans.dw;
    free(ans.dcam.aligned);
    free(ans.dX.aligned);

    insert_reproj_err_block(J, i, camIdx, ptIdx, reproj_err_d);
  }
}

void calculate_w_jacobian(BAApp app, BAInput input, BASparseMat *J) {
  for (size_t j = 0; j < input.p; j++) {
    double res = app.w_func(input.w[j]);
    insert_w_err_block(J, j, res);
  }
}

unsigned long compute_jacobian(BAApp app, BAInput input, BASparseMat *mat,
                               BASparseMat *ref) {
  struct timeval start, stop;
  clearBASparseMat(mat);
  gettimeofday(&start, NULL);
  calculate_reproj_jacobian(app, input, mat);
  calculate_w_jacobian(app, input, mat);
  gettimeofday(&stop, NULL);
  verify_ba_results(ref, mat, app.name);
  return timediff(start, stop);
}

void populate_ref(BAApp app, BAInput ba_input, BASparseMat *ref) {
  clearBASparseMat(ref);
  calculate_reproj_jacobian(app, ba_input, ref);
  calculate_w_jacobian(app, ba_input, ref);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <data_file>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  BAInput ba_input = read_ba_data(argv[1]);
  int n = ba_input.n, m = ba_input.m, p = ba_input.p;
  BASparseMat mat = initBASparseMat(n, m, p);
  BASparseMat ref = initBASparseMat(n, m, p);

  BAApp apps[] = {{.name = "LAGrad",
                   .reproj_func = lagrad_compute_reproj_error_wrapper,
                   .w_func = lagrad_compute_w_error},
                  {.name = "Enzyme/C",
                   .reproj_func = enzyme_c_compute_reproj_err,
                   .w_func = enzyme_c_compute_w_err},
                  {.name = "Enzyme/MLIR",
                   .reproj_func = enzyme_compute_reproj_error_wrapper,
                   .w_func = enzyme_compute_w_error}};
  size_t num_apps = sizeof(apps) / sizeof(apps[0]);

  populate_ref(apps[0], ba_input, &ref);
  unsigned long results_df[NUM_RUNS];
  for (size_t app = 0; app < num_apps; app++) {
    printf("%s: ", apps[app].name);
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = compute_jacobian(apps[app], ba_input, &mat, &ref);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free_ba_data(ba_input);
  freeBASparseMat(&mat);
  freeBASparseMat(&ref);
}
