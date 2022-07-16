#include "ba.h"
#include "mlir_c_abi.h"

#define NUM_RUNS 6

double *deadbeef = (double *)0xdeadbeef;

extern BAGrad enzyme_compute_reproj_error(
    /*cam=*/double *, double *, int64_t, int64_t, int64_t,
    /*X==*/double *, double *, int64_t, int64_t, int64_t,
    /*w=*/double,
    /*feat==*/double *, double *, int64_t, int64_t, int64_t,
    /*g=*/double *, double *, int64_t, int64_t, int64_t);

extern double enzyme_compute_w_error(double w);

extern BAGrad lagrad_compute_reproj_error(
    /*cam=*/double *, double *, int64_t, int64_t, int64_t,
    /*X==*/double *, double *, int64_t, int64_t, int64_t,
    /*w=*/double,
    /*feat==*/double *, double *, int64_t, int64_t, int64_t,
    /*g=*/double *, double *, int64_t, int64_t, int64_t);

extern double lagrad_compute_w_error(double w);

extern void dcompute_reproj_error(double const *cam, double *dcam,
                                  double const *X, double *dX, double const *w,
                                  double *wb, double const *feat, double *err,
                                  double *derr);

extern void dcompute_w_error(double const *w, double *wb, double *err,
                             double *derr);

void enzyme_c_calculate_reproj_jacobian(BAInput ba_input, BASparseMat *J) {
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
    dcompute_reproj_error(&ba_input.cams[camIdx * BA_NCAMPARAMS], dcam,
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
    dcompute_reproj_error(&ba_input.cams[camIdx * BA_NCAMPARAMS], dcam,
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

void enzyme_c_calculate_w_jacobian(BAInput input, BASparseMat *J) {
  for (size_t j = 0; j < input.p; j++) {
    double wb = 0.0;
    double err = 0.0;
    double errb = 1.0;
    dcompute_w_error(&input.w[j], &wb, &err, &errb);
    insert_w_err_block(J, j, wb);
  }
}

void enzyme_calculate_reproj_jacobian(BAInput ba_input, BASparseMat *J) {
  double errb[2];
  double reproj_err_d[2 * (BA_NCAMPARAMS + 3 + 1)];
  for (size_t i = 0; i < ba_input.p; i++) {
    int camIdx = ba_input.obs[2 * i + 0];
    int ptIdx = ba_input.obs[2 * i + 1];

    // Calculate first row
    errb[0] = 1.0;
    errb[1] = 0.0;

    BAGrad ans = enzyme_compute_reproj_error(
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

    // Calculate second row
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

void enzyme_calculate_w_jacobian(BAInput input, BASparseMat *J) {
  for (size_t j = 0; j < input.p; j++) {
    double res = enzyme_compute_w_error(input.w[j]);
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

unsigned long enzyme_c_compute_jacobian(BAInput input, BASparseMat *mat,
                                        BASparseMat *ref) {
  struct timeval start, stop;
  clearBASparseMat(mat);
  gettimeofday(&start, NULL);
  enzyme_c_calculate_reproj_jacobian(input, mat);
  enzyme_c_calculate_w_jacobian(input, mat);
  gettimeofday(&stop, NULL);
  verify_ba_results(ref, mat, "Enzyme/C");
  return timediff(start, stop);
}

unsigned long enzyme_compute_jacobian(BAInput input, BASparseMat *mat,
                                      BASparseMat *ref) {
  struct timeval start, stop;
  clearBASparseMat(mat);
  gettimeofday(&start, NULL);
  enzyme_calculate_reproj_jacobian(input, mat);
  enzyme_calculate_w_jacobian(input, mat);
  gettimeofday(&stop, NULL);
  verify_ba_results(ref, mat, "Enzyme/MLIR");
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

int generate_main() {
  BAInput ba_input = read_ba_data("{{data_file}}");
  int n = ba_input.n, m = ba_input.m, p = ba_input.p;
  BASparseMat mat = initBASparseMat(n, m, p);

  clearBASparseMat(&mat);
  lagrad_calculate_reproj_jacobian(ba_input, &mat);
  lagrad_calculate_w_jacobian(ba_input, &mat);
  serialize_sparse_mat("{{results_file}}", &mat);

  free_ba_data(ba_input);
  freeBASparseMat(&mat);
  return 0;
}

extern F64Descriptor1D mlir_compute_reproj_error(/*cam=*/double *, double *,
                                                 int64_t, int64_t, int64_t,
                                                 /*X==*/double *, double *,
                                                 int64_t, int64_t, int64_t,
                                                 /*w=*/double,
                                                 /*feat==*/double *, double *,
                                                 int64_t, int64_t, int64_t);
extern void ecompute_reproj_error(double const *__restrict cam,
                                  double const *__restrict X,
                                  double const *__restrict w,
                                  double const *__restrict feat,
                                  double *__restrict err);

int test_main() {
  BAInput ba_input = read_ba_data("benchmarks/data/ba/test.txt");
  int n = ba_input.n, m = ba_input.m, p = ba_input.p;
  BASparseMat mat = initBASparseMat(n, m, p);

  clearBASparseMat(&mat);
  enzyme_c_calculate_reproj_jacobian(ba_input, &mat);
  enzyme_c_calculate_w_jacobian(ba_input, &mat);
  print_d_arr(mat.vals, 10);
  return 0;
  // enzyme_c_calculate_reproj_jacobian(ba_input, &mat);
  // enzyme_c_calculate_w_jacobian(ba_input, &mat);
  int i = 6;
  int camIdx = ba_input.obs[2 * i + 0];
  int ptIdx = ba_input.obs[2 * i + 1];

  F64Descriptor1D mlir_primal = mlir_compute_reproj_error(
      /*cams=*/deadbeef, &ba_input.cams[camIdx * BA_NCAMPARAMS], 0,
      BA_NCAMPARAMS, 1,
      /*X=*/deadbeef, &ba_input.X[ptIdx * 3], 0, 3, 1, /*w=*/ba_input.w[i],
      deadbeef, &ba_input.feats[i * 2], 0, 2, 1);
  printf("MLIR primal:\n");
  print_d_arr(mlir_primal.aligned, mlir_primal.size);
  double err[2] = {0, 0};
  ecompute_reproj_error(&ba_input.cams[camIdx * BA_NCAMPARAMS],
                        &ba_input.X[ptIdx * 3], &ba_input.w[i],
                        &ba_input.feats[i * 2], err);
  printf("Enzyme/C primal:\n");
  print_d_arr(err, 2);
  // ecompute_reproj_error(ba_input.cams)

  free_ba_data(ba_input);
  // freeBASparseMat(&mat);
  return 0;
}

void populate_ref(BAInput ba_input, BASparseMat *ref) {
  clearBASparseMat(ref);
  enzyme_calculate_reproj_jacobian(ba_input, ref);
  enzyme_calculate_w_jacobian(ba_input, ref);
}

int main() {
  BAInput ba_input = read_ba_data("{{data_file}}");
  int n = ba_input.n, m = ba_input.m, p = ba_input.p;
  BASparseMat mat = initBASparseMat(n, m, p);
  BASparseMat ref = initBASparseMat(n, m, p);
  populate_ref(ba_input, &ref);

  bodyFunc funcs[] = {lagrad_compute_jacobian, enzyme_compute_jacobian,
                      enzyme_c_compute_jacobian};
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);
  unsigned long *results_df =
      (unsigned long *)malloc(NUM_RUNS * sizeof(unsigned long));

  for (size_t app = 0; app < num_apps; app++) {
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = (*funcs[app])(ba_input, &mat, &ref);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }
  free_ba_data(ba_input);
  freeBASparseMat(&mat);
  freeBASparseMat(&ref);
  free(results_df);
  return 0;
}
