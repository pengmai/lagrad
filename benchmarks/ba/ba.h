#pragma once
#include "lagrad_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BA_NCAMPARAMS 11

typedef struct _BAInput {
  int n, m, p;
  double *cams, *X, *w, *feats;
  int *obs;
} BAInput;

typedef struct _BAGrad {
  F64Descriptor1D dcam;
  F64Descriptor1D dX;
  double dw;
} BAGrad;

typedef struct _BASparseMat {
  int n, m, p;
  int nrows, ncols;
  int row_end, col_end, val_end;
  int *rows;
  int *cols;
  double *vals;
} BASparseMat;

BAGrad enzyme_compute_reproj_error(
    /*cam=*/double *, double *, int64_t, int64_t, int64_t,
    /*X==*/double *, double *, int64_t, int64_t, int64_t,
    /*w=*/double,
    /*feat==*/double *, double *, int64_t, int64_t, int64_t,
    /*g=*/double *, double *, int64_t, int64_t, int64_t);

double enzyme_compute_w_error(double w);

BAGrad lagrad_compute_reproj_error(
    /*cam=*/double *, double *, int64_t, int64_t, int64_t,
    /*X==*/double *, double *, int64_t, int64_t, int64_t,
    /*w=*/double,
    /*feat==*/double *, double *, int64_t, int64_t, int64_t,
    /*g=*/double *, double *, int64_t, int64_t, int64_t);

double lagrad_compute_w_error(double w);

extern void dcompute_reproj_error(double const *cam, double *dcam,
                                  double const *X, double *dX, double const *w,
                                  double *wb, double const *feat, double *err,
                                  double *derr);

extern void dcompute_w_error(double const *w, double *wb, double *err,
                             double *derr);

void insert_reproj_err_block(BASparseMat *mat, int obsIdx, int camIdx,
                             int ptIdx, const double *J) {
  int n_new_cols = BA_NCAMPARAMS + 3 + 1;
  mat->rows[mat->row_end] = mat->rows[mat->row_end - 1] + n_new_cols;
  mat->row_end++;
  mat->rows[mat->row_end] = mat->rows[mat->row_end - 1] + n_new_cols;
  mat->row_end++;

  for (int i_row = 0; i_row < 2; i_row++) {
    for (int i = 0; i < BA_NCAMPARAMS; i++) {
      mat->cols[mat->col_end] = BA_NCAMPARAMS * camIdx + i;
      mat->col_end++;
      mat->vals[mat->val_end] = J[2 * i + i_row];
      mat->val_end++;
    }
    int col_offset = BA_NCAMPARAMS * mat->n;
    int val_offset = BA_NCAMPARAMS * 2;
    for (int i = 0; i < 3; i++) {
      mat->cols[mat->col_end] = col_offset + 3 * ptIdx + i;
      mat->col_end++;
      mat->vals[mat->val_end] = J[val_offset + 2 * i + i_row];
      mat->val_end++;
    }
    col_offset += 3 * mat->m;
    val_offset += 3 * 2;
    mat->cols[mat->col_end] = col_offset + obsIdx;
    mat->col_end++;
    mat->vals[mat->val_end] = J[val_offset + i_row];
    mat->val_end++;
  }
}

void insert_w_err_block(BASparseMat *mat, int wIdx, double w_d) {
  mat->rows[mat->row_end] = mat->rows[mat->row_end - 1] + 1;
  mat->row_end++;
  mat->cols[mat->col_end] = BA_NCAMPARAMS * mat->n + 3 * mat->m + wIdx;
  mat->col_end++;
  mat->vals[mat->val_end] = w_d;
  mat->val_end++;
}

BASparseMat initBASparseMat(int n, int m, int p) {
  int nrows = 2 * p + p;
  int ncols = BA_NCAMPARAMS * n + 3 * m + p;
  int nnonzero = (BA_NCAMPARAMS + 3 + 1) * 2 * p + p;
  int *rows = (int *)malloc((nrows + 1) * sizeof(int));
  rows[0] = 0;
  int *cols = (int *)malloc(nnonzero * sizeof(int));
  double *vals = (double *)malloc(nnonzero * sizeof(double));
  BASparseMat mat = {.n = n,
                     .m = m,
                     .p = p,
                     .nrows = nrows,
                     .ncols = ncols,
                     .row_end = 1,
                     .col_end = 0,
                     .val_end = 0,
                     .rows = rows,
                     .cols = cols,
                     .vals = vals};
  return mat;
}

void freeBASparseMat(BASparseMat *mat) {
  free(mat->rows);
  free(mat->cols);
  free(mat->vals);
}

void clearBASparseMat(BASparseMat *mat) {
  mat->rows[0] = 0;
  mat->row_end = 1;
  mat->col_end = 0;
  mat->val_end = 0;
}

BAInput read_ba_data(const char *data_file) {
  FILE *fp = fopen(data_file, "r");
  if (!fp) {
    fprintf(stderr, "Failed to open file \"%s\"\n", data_file);
    exit(EXIT_FAILURE);
  }

  int n, m, p;
  fscanf(fp, "%d %d %d", &n, &m, &p);
  int nCamParams = 11;

  double *cams = (double *)malloc(nCamParams * n * sizeof(double));
  double *X = (double *)malloc(3 * m * sizeof(double));
  double *w = (double *)malloc(p * sizeof(double));
  double *feats = (double *)malloc(2 * p * sizeof(double));
  int *obs = (int *)malloc(2 * p * sizeof(int));

  for (size_t j = 0; j < nCamParams; j++) {
    fscanf(fp, "%lf", &cams[j]);
  }
  for (size_t i = 1; i < n; i++) {
    memcpy(&cams[i * nCamParams], &cams[0], nCamParams * sizeof(double));
  }

  for (size_t j = 0; j < 3; j++) {
    fscanf(fp, "%lf", &X[j]);
  }
  for (size_t i = 0; i < m; i++) {
    memcpy(&X[i * 3], &X[0], 3 * sizeof(double));
  }

  fscanf(fp, "%lf", &w[0]);
  for (size_t i = 1; i < p; i++) {
    w[i] = w[0];
  }

  int camIdx = 0;
  int ptIdx = 0;
  for (size_t i = 0; i < p; i++) {
    obs[i * 2 + 0] = (camIdx++ % n);
    obs[i * 2 + 1] = (ptIdx++ % m);
  }

  fscanf(fp, "%lf %lf", &feats[0], &feats[1]);
  for (size_t i = 1; i < p; i++) {
    feats[i * 2 + 0] = feats[0];
    feats[i * 2 + 1] = feats[1];
  }

  BAInput ba_input = {.n = n,
                      .m = m,
                      .p = p,
                      .cams = cams,
                      .X = X,
                      .w = w,
                      .feats = feats,
                      .obs = obs};
  fclose(fp);
  return ba_input;
}

void verify_ba_results(BASparseMat *ref, BASparseMat *actual, const char *app) {
  int p = actual->p;
  int nrows = 2 * p + p;
  int nnonzero = (BA_NCAMPARAMS + 3 + 1) * 2 * p + p;
  int row_size = nrows + 1;

  for (size_t i = 0; i < row_size; i++) {
    if (ref->rows[i] != actual->rows[i]) {
      printf("(%s) Incorrect row val at index %lu: %d\n", app, i,
             actual->rows[i]);
    }
  }

  for (size_t i = 0; i < nnonzero; i++) {
    if (ref->cols[i] != actual->cols[i]) {
      printf("(%s) Incorrect col val at index %lu: %d\n", app, i,
             actual->cols[i]);
    }
  }

  double max_err = 0.0;
  for (size_t i = 0; i < nnonzero; i++) {
    double err = fabs(ref->vals[i] - actual->vals[i]);
    if (err > max_err) {
      max_err = err;
    }
  }
  if (max_err > 1e-6) {
    printf("(%s) Vals max err: %f\n", app, max_err);
  }
}

void free_ba_data(BAInput ba_input) {
  free(ba_input.cams);
  free(ba_input.X);
  free(ba_input.w);
  free(ba_input.feats);
  free(ba_input.obs);
}
