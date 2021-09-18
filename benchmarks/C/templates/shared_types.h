#pragma once

typedef struct {
  float *da;
  float *db;
} RawDotGradient;

typedef struct {
  float *da;
  float *db;
} F32PTuple;

typedef struct _GMMInput {
  int d, k, n;
  double *alphas, *means, *Qs, *Ls, *x;
  double wishart_gamma;
  int wishart_m;
} GMMInput;
