#include "ba.h"
#include "mlir_c_abi.h"

#define NUM_RUNS 1

double *deadbeef = (double *)0xdeadbeef;

extern void enzyme_c_compute_reproj_error(double const *cam, double *dcam,
                                          double const *X, double *dX,
                                          double const *w, double *wb,
                                          double const *feat, double *err,
                                          double *derr);
extern void enzyme_c_compute_w_error(double const *w, double *wb, double *err,
                                     double *derr);
extern BAGrad enzyme_compute_reproj_error(
    /*cam=*/double *, double *, int64_t, int64_t, int64_t,
    /*X==*/double *, double *, int64_t, int64_t, int64_t,
    /*w=*/double,
    /*feat==*/double *, double *, int64_t, int64_t, int64_t,
    /*g=*/double *, double *, int64_t, int64_t, int64_t);

extern double enzyme_compute_w_error(double w);

extern double lagrad_compute_reproj_error(
    /*cam=*/double *, double *, int64_t, int64_t, int64_t,
    /*dcam=*/double *, double *, int64_t, int64_t, int64_t,
    /*X=*/double *, double *, int64_t, int64_t, int64_t,
    /*dX=*/double *, double *, int64_t, int64_t, int64_t,
    /*w=*/double,
    /*dw=*/double,
    /*feat=*/double *, double *, int64_t, int64_t, int64_t,
    /*g=*/double *, double *, int64_t, int64_t, int64_t);

/* Primal declarations */
extern void mlir_compute_reproj_error(
    /*cam=*/double *, double *, int64_t, int64_t, int64_t,
    /*X=*/double *, double *, int64_t, int64_t, int64_t,
    /*w=*/double,
    /*feat=*/double *, double *, int64_t, int64_t, int64_t,
    /*out=*/double *, double *, int64_t, int64_t, int64_t);

extern void ecompute_reproj_error(double const *__restrict cam,
                                  double const *__restrict X,
                                  double const *__restrict w,
                                  double const *__restrict feat,
                                  double *__restrict err);

extern double lagrad_compute_w_error(double w);

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
    ans = enzyme_compute_reproj_error(
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

void enzyme_c_calculate_reproj_jacobian(BAInput ba_input, BASparseMat *J) {
  double errb[2];
  double err[2];
  double reproj_err_d[2 * (BA_NCAMPARAMS + 3 + 1)];
  double *reproj_err_b =
      (double *)malloc((BA_NCAMPARAMS + 3 + 1) * sizeof(double));
  for (size_t i = 0; i < ba_input.p; i++) {
    int camIdx = ba_input.obs[2 * i + 0];
    int ptIdx = ba_input.obs[2 * i + 1];

    // Calculate first row.
    err[0] = err[1] = 0;
    errb[0] = 1.0;
    errb[1] = 0.0;
    for (size_t j = 0; j < BA_NCAMPARAMS + 3 + 1; j++) {
      reproj_err_b[j] = 0;
    }
    enzyme_c_compute_reproj_error(
        &ba_input.cams[camIdx * BA_NCAMPARAMS], reproj_err_b,
        &ba_input.X[ptIdx * 3], reproj_err_b + BA_NCAMPARAMS, &ba_input.w[i],
        reproj_err_b + BA_NCAMPARAMS + 3, &ba_input.feats[i * 2], err, errb);

    for (size_t j = 0; j < BA_NCAMPARAMS + 3 + 1; j++) {
      reproj_err_d[2 * j] = reproj_err_b[j];
    }

    err[0] = err[1] = 0;
    errb[0] = 0.0;
    errb[1] = 1.0;
    for (size_t j = 0; j < BA_NCAMPARAMS + 3 + 1; j++) {
      reproj_err_b[j] = 0;
    }

    enzyme_c_compute_reproj_error(
        &ba_input.cams[camIdx * BA_NCAMPARAMS], reproj_err_b,
        &ba_input.X[ptIdx * 3], reproj_err_b + BA_NCAMPARAMS, &ba_input.w[i],
        reproj_err_b + BA_NCAMPARAMS + 3, &ba_input.feats[i * 2], err, errb);
    for (size_t j = 0; j < BA_NCAMPARAMS + 3 + 1; j++) {
      reproj_err_d[2 * j + 1] = reproj_err_b[j];
    }

    insert_reproj_err_block(J, i, camIdx, ptIdx, reproj_err_d);
  }
  free(reproj_err_b);
}

void enzyme_c_calculate_w_jacobian(BAInput input, BASparseMat *J) {
  for (size_t j = 0; j < input.p; j++) {
    double wb = 0.0;
    double err = 0.0;
    double errb = 1.0;
    enzyme_c_compute_w_error(&input.w[j], &wb, &err, &errb);
    insert_w_err_block(J, j, wb);
  }
}

void lagrad_calculate_reproj_jacobian(BAInput ba_input, BASparseMat *J) {
  double errb[2];
  double reproj_err_d[2 * (BA_NCAMPARAMS + 3 + 1)];
  double reproj_err_b[BA_NCAMPARAMS + 3];
  for (size_t i = 0; i < ba_input.p; i++) {
    int camIdx = ba_input.obs[2 * i + 0];
    int ptIdx = ba_input.obs[2 * i + 1];

    // Calculate first row
    errb[0] = 1.0;
    errb[1] = 0.0;

    for (size_t i = 0; i < BA_NCAMPARAMS + 3; i++) {
      reproj_err_b[i] = 0;
    }

    double dw = lagrad_compute_reproj_error(
        /*cams=*/deadbeef, &ba_input.cams[camIdx * BA_NCAMPARAMS], 0,
        BA_NCAMPARAMS, 1,
        /*dcams=*/deadbeef, reproj_err_b, 0, BA_NCAMPARAMS, 1,
        /*X=*/deadbeef, &ba_input.X[ptIdx * 3], 0, 3, 1,
        /*dX=*/deadbeef, reproj_err_b + BA_NCAMPARAMS, 0, 3, 1,
        /*w=*/ba_input.w[i], /*dw=*/0,
        /*feats=*/deadbeef, &ba_input.feats[i * 2], 0, 2, 1,
        /*g=*/deadbeef, errb, 0, 2, 1);
    for (size_t j = 0; j < BA_NCAMPARAMS + 3; j++) {
      reproj_err_d[2 * j] = reproj_err_b[j];
    }
    reproj_err_d[2 * (BA_NCAMPARAMS + 3)] = dw;

    errb[1] = 0.0;
    errb[0] = 1.0;
    for (size_t i = 0; i < BA_NCAMPARAMS + 3; i++) {
      reproj_err_b[i] = 0;
    }
    dw = lagrad_compute_reproj_error(
        /*cams=*/deadbeef, &ba_input.cams[camIdx * BA_NCAMPARAMS], 0,
        BA_NCAMPARAMS, 1,
        /*dcams=*/deadbeef, reproj_err_b, 0, BA_NCAMPARAMS, 1,
        /*X=*/deadbeef, &ba_input.X[ptIdx * 3], 0, 3, 1,
        /*dX=*/deadbeef, reproj_err_b + BA_NCAMPARAMS, 0, 3, 1,
        /*w=*/ba_input.w[i], /*dw=*/0,
        /*feats=*/deadbeef, &ba_input.feats[i * 2], 0, 2, 1,
        /*g=*/deadbeef, errb, 0, 2, 1);
    for (size_t j = 0; j < BA_NCAMPARAMS + 3; j++) {
      reproj_err_d[2 * j + 1] = reproj_err_b[j];
    }
    reproj_err_d[2 * (BA_NCAMPARAMS + 3) + 1] = dw;

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
  // verify_ba_results(ref, mat, "Enzyme/C");
  printf("Enzyme\\C\n");
  printf("nrows: %d, ncols: %d\n", mat->nrows, mat->ncols);
  printf("row end: %d, col end: %d, vals: %d\n", mat->row_end, mat->col_end,
         mat->val_end);
  // printf("First ten values:\n");
  // print_d_arr(mat->vals);
  FILE *fp = fopen("ba_testvals.txt", "w");
  if (fp == NULL) {
    printf("Failed to open file\n");
    exit(EXIT_FAILURE);
  }
  fprintf(fp, "[");
  for (size_t i = 0; i < mat->val_end; i++) {
    fprintf(fp, "%lf", mat->vals[i]);
    if (i != mat->val_end - 1) {
      fprintf(fp, ", ");
    }
  }
  fprintf(fp, "]\n");
  fclose(fp);

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
  // verify_ba_results(ref, mat, "LAGrad");
  // printf("nrows: %d, ncols: %d\n", mat->nrows, mat->ncols);
  // printf("row end: %d, col end: %d, vals: %d\n", mat->row_end, mat->col_end,
  //        mat->val_end);
  // printf("First ten values:\n");
  // print_d_arr(mat->vals, 10);
  return timediff(start, stop);
}

void dothething(BAInput ba_input) {
  int i = 0;
  int camIdx = ba_input.obs[2 * i + 0];
  int ptIdx = ba_input.obs[2 * i + 1];
  double reproj_err_b[BA_NCAMPARAMS + 3 + 1];
  double errb[] = {1., 1.};
  for (size_t i = 0; i < BA_NCAMPARAMS + 3 + 1; i++) {
    reproj_err_b[i] = 0;
  }

  // double dw = lagrad_compute_reproj_error(
  //     /*cams=*/deadbeef, &ba_input.cams[camIdx * BA_NCAMPARAMS], 0,
  //     BA_NCAMPARAMS, 1,
  //     /*dcams=*/deadbeef, reproj_err_b, 0, BA_NCAMPARAMS, 1,
  //     /*X=*/deadbeef, &ba_input.X[ptIdx * 3], 0, 3, 1,
  //     /*dX=*/deadbeef, reproj_err_b + BA_NCAMPARAMS, 0, 3, 1,
  //     /*w=*/ba_input.w[i], /*dw=*/0,
  //     /*feats=*/deadbeef, &ba_input.feats[i * 2], 0, 2, 1,
  //     /*g=*/deadbeef, errb, 0, 2, 1);
  // printf("LAGrad dw: %f\n", dw);
  // print_d_arr(reproj_err_b, BA_NCAMPARAMS + 3);

  uniform_init_d(0.0, reproj_err_b, BA_NCAMPARAMS + 3 + 1);
  double err[] = {0., 0.};
  for (size_t i = 0; i < 2; i++) {
    errb[i] = 1.0;
  }

  // enzyme_c_compute_reproj_error(
  //     &ba_input.cams[camIdx * BA_NCAMPARAMS], reproj_err_b,
  //     &ba_input.X[ptIdx * 3], reproj_err_b + BA_NCAMPARAMS, &ba_input.w[i],
  //     reproj_err_b + BA_NCAMPARAMS + 3, &ba_input.feats[i * 2], err, errb);
  // printf("Enzyme dw: %f\n", reproj_err_b[BA_NCAMPARAMS + 3]);
  // print_d_arr(reproj_err_b, BA_NCAMPARAMS + 3);

  // primals
  uniform_init_d(0.0, err, 2);
  ecompute_reproj_error(ba_input.cams, ba_input.X, ba_input.w, ba_input.feats,
                        err);
  printf("C primal err: %f %f\n", err[0], err[1]);

  uniform_init_d(0.0, err, 2);
  mlir_compute_reproj_error(
      /*cams=*/deadbeef, &ba_input.cams[camIdx * BA_NCAMPARAMS], 0,
      BA_NCAMPARAMS, 1,
      /*X=*/deadbeef, &ba_input.X[ptIdx * 3], 0, 3, 1,
      /*w=*/ba_input.w[i],
      /*feats=*/deadbeef, &ba_input.feats[i * 2], 0, 2, 1,
      /*out=*/deadbeef, err, 0, 2, 1);
  printf("MLIR primal err: %f %f\n", err[0], err[1]);
  printf("Done\n");
}

int main() {
  BAInput ba_input = read_ba_data();
  int n = ba_input.n, m = ba_input.m, p = ba_input.p;
  BASparseMat mat = initBASparseMat(n, m, p);
  BASparseMat ref = initBASparseMat(n, m, p);
  read_ba_results(&ref);

  dothething(ba_input);

  // bodyFunc funcs[] = {lagrad_compute_jacobian,
  //                     // enzyme_compute_jacobian,
  //                     enzyme_c_compute_jacobian};
  // size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);
  // unsigned long *results_df =
  //     (unsigned long *)malloc(NUM_RUNS * sizeof(unsigned long));

  // for (size_t app = 0; app < num_apps; app++) {
  //   for (size_t run = 0; run < NUM_RUNS; run++) {
  //     results_df[run] = (*funcs[app])(ba_input, &mat, &ref);
  //   }
  //   print_ul_arr(results_df, NUM_RUNS);
  // }
  free_ba_data(ba_input);
  freeBASparseMat(&mat);
  freeBASparseMat(&ref);
  // free(results_df);
}
