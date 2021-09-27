#pragma once
#include "mlir_c_abi.h"

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
  double *alphas, *means, *Qs, *Ls, *x, *icf;
  double wishart_gamma;
  int wishart_m;
} GMMInput;

typedef struct _GMMGrad {
  F64Descriptor1D dalphas;
  F64Descriptor2D dmeans, dqs;
  F64Descriptor3D dls;
} GMMGrad;
