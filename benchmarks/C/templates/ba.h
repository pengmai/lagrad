#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BA_DATA_FILE "benchmarks/data/ba1_n49_m7776_p31843.txt"

typedef struct _BAInput {
  int n, m, p;
  double *cams, *X, *w, *feats;
  int *obs;
} BAInput;

BAInput read_ba_data() {
  FILE *fp = fopen(BA_DATA_FILE, "r");
  if (!fp) {
    fprintf(stderr, "Failed to open file \"%s\"\n", BA_DATA_FILE);
    exit(EXIT_FAILURE);
  }

  int n, m, p;
  fscanf(fp, "%d %d %d", &n, &m, &p);
  // Not sure where this comes from, copied from Enzyme
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

void free_ba_data(BAInput ba_input) {
  free(ba_input.cams);
  free(ba_input.X);
  free(ba_input.w);
  free(ba_input.feats);
  free(ba_input.obs);
}
