#include "ba.h"
#include "mlir_c_abi.h"

#define NUM_RUNS 10

double *deadbeef = (double *)0xdeadbeef;

extern void enzyme_compute_reproj_error(double const *cam, double *dcam,
                                        double *X, double *dX, double *w,
                                        double *wb, double *feat, double *err,
                                        double *derr);

extern void enzyme_compute_w_error(double *w, double *dw, double *err,
                                   double *derr);

extern BAGrad lagrad_compute_reproj_error(
    /*cam=*/double *, double *, int64_t, int64_t, int64_t,
    /*X==*/double *, double *, int64_t, int64_t, int64_t,
    /*w=*/double,
    /*feat==*/double *, double *, int64_t, int64_t, int64_t,
    /*g=*/double *, double *, int64_t, int64_t, int64_t);

extern double lagrad_compute_w_error(double w);

void enzyme_calculate_reproj_jacobian(BAInput ba_input, BASparseMat *J) {
  double errb[2];
  double err[2] = {0.0, 0.0};
  double reproj_err_d[2 * (BA_NCAMPARAMS + 3 + 1)];
  for (size_t i = 0; i < ba_input.p; i++) {
    int camIdx = ba_input.obs[2 * i + 0];
    int ptIdx = ba_input.obs[2 * i + 1];

    // Calculate first row
    err[0] = err[1] = 0;
    errb[0] = 1.0;
    errb[1] = 0.0;

    double *dcam = (double *)malloc(BA_NCAMPARAMS * sizeof(double));
    for (size_t i = 0; i < BA_NCAMPARAMS; i++) {
      dcam[i] = 0;
    }
    double *dX = (double *)malloc(3 * sizeof(double));
    for (size_t i = 0; i < 3; i++) {
      dX[i] = 0;
    }
    double wb = 0.0;
    enzyme_compute_reproj_error(&ba_input.cams[camIdx * BA_NCAMPARAMS], dcam,
                                &ba_input.X[ptIdx * 3], dX, &ba_input.w[i], &wb,
                                &ba_input.feats[i * 2],
                                /*g=*/err, errb);
    size_t j = 0;
    for (j = 0; j < BA_NCAMPARAMS; j++) {
      reproj_err_d[j * 2] = dcam[j];
    }
    for (size_t k = 0; k < 3; k++) {
      reproj_err_d[j * 2] = dX[k];
      j++;
    }
    reproj_err_d[j * 2] = wb;
    free(dcam);
    free(dX);

    err[0] = err[1] = 0;
    errb[1] = 0.0;
    errb[0] = 1.0;

    dcam = (double *)malloc(BA_NCAMPARAMS * sizeof(double));
    for (size_t i = 0; i < BA_NCAMPARAMS; i++) {
      dcam[i] = 0;
    }
    dX = (double *)malloc(3 * sizeof(double));
    for (size_t i = 0; i < 3; i++) {
      dX[i] = 0;
    }
    wb = 0.0;
    enzyme_compute_reproj_error(&ba_input.cams[camIdx * BA_NCAMPARAMS], dcam,
                                &ba_input.X[ptIdx * 3], dX, &ba_input.w[i], &wb,
                                &ba_input.feats[i * 2],
                                /*g=*/err, errb);
    j = 0;
    for (j = 0; j < BA_NCAMPARAMS; j++) {
      reproj_err_d[j * 2 + 1] = dcam[j];
    }
    for (size_t k = 0; k < 3; k++) {
      reproj_err_d[j * 2 + 1] = dX[k];
      j++;
    }
    reproj_err_d[j * 2 + 1] = wb;
    free(dcam);
    free(dX);

    insert_reproj_err_block(J, i, camIdx, ptIdx, reproj_err_d);
  }
}

void enzyme_calculate_w_jacobian(BAInput input, BASparseMat *J) {
  for (size_t j = 0; j < input.p; j++) {
    double res, err, errb;
    res = 0;
    err = 0;
    errb = 1.0;
    enzyme_compute_w_error(&input.w[j], &res, &err, &errb);
    insert_w_err_block(J, j, res);
  }
}

void lagrad_calculate_reproj_jacobian(BAInput ba_input, BASparseMat *J) {
  double errb[2];
  double reproj_err_d[2 * (BA_NCAMPARAMS + 3 + 1)];
  for (size_t i = 0; i < ba_input.p; i++) {
    int camIdx = ba_input.obs[2 * i + 0];
    int ptIdx = ba_input.obs[2 * i + 1];

    // Calculate first row
    errb[0] = 1.0;
    errb[1] = 0.0;

    BAGrad ans = lagrad_compute_reproj_error(
        /*cams=*/deadbeef, &ba_input.cams[camIdx * BA_NCAMPARAMS], 0,
        BA_NCAMPARAMS, 1,
        /*X=*/deadbeef, &ba_input.X[ptIdx * 3], 0, 3, 1, /*w=*/ba_input.w[i],
        deadbeef, &ba_input.feats[i * 2], 0, 2, 1,
        /*g=*/deadbeef, errb, 0, 2, 1);
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
    ans = lagrad_compute_reproj_error(
        /*cams=*/deadbeef, &ba_input.cams[camIdx * BA_NCAMPARAMS], 0,
        BA_NCAMPARAMS, 1,
        /*X=*/deadbeef, &ba_input.X[ptIdx * 3], 0, 3, 1, /*w=*/ba_input.w[i],
        deadbeef, &ba_input.feats[i * 2], 0, 2, 1,
        /*g=*/deadbeef, errb, 0, 2, 1);
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

void lagrad_calculate_w_jacobian(BAInput input, BASparseMat *J) {
  for (size_t j = 0; j < input.p; j++) {
    double res = lagrad_compute_w_error(input.w[j]);
    insert_w_err_block(J, j, res);
  }
}

typedef unsigned long (*bodyFunc)(BAInput input, BASparseMat *mat,
                                  BASparseMat *ref);

unsigned long enzyme_compute_jacobian(BAInput input, BASparseMat *mat,
                                      BASparseMat *ref) {
  struct timeval start, stop;
  clearBASparseMat(mat);
  gettimeofday(&start, NULL);
  enzyme_calculate_reproj_jacobian(input, mat);
  enzyme_calculate_w_jacobian(input, mat);
  gettimeofday(&stop, NULL);
  verify_ba_results(ref, mat, "Enzyme C");
  return timediff(start, stop);
}

unsigned long lagrad_compute_jacobian(BAInput input, BASparseMat *mat,
                                      BASparseMat *ref) {
  struct timeval start, stop;
  clearBASparseMat(mat);
  gettimeofday(&start, NULL);
  lagrad_calculate_reproj_jacobian(input, mat);
  lagrad_calculate_w_jacobian(input, mat);
  gettimeofday(&stop, NULL);
  verify_ba_results(ref, mat, "LAGrad");
  return timediff(start, stop);
}

int main() {
  BAInput ba_input = read_ba_data();
  int n = ba_input.n, m = ba_input.m, p = ba_input.p;
  BASparseMat mat = initBASparseMat(n, m, p);
  BASparseMat ref = initBASparseMat(n, m, p);
  read_ba_results(&ref);

  bodyFunc funcs[] = {enzyme_compute_jacobian, lagrad_compute_jacobian};
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);
  unsigned long *results_df =
      (unsigned long *)malloc(num_apps * NUM_RUNS * sizeof(unsigned long));

  for (size_t app = 0; app < num_apps; app++) {
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[app * NUM_RUNS + run] = (*funcs[app])(ba_input, &mat, &ref);
    }
    print_ul_arr(results_df + app * NUM_RUNS, NUM_RUNS);
  }
  free_ba_data(ba_input);
  freeBASparseMat(&mat);
  freeBASparseMat(&ref);
  free(results_df);
}
