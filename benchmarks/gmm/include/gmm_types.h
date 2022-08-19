#pragma once
#include "lagrad_utils.h"

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

typedef struct GMMCompressedGrad {
  F64Descriptor1D dalphas;
  F64Descriptor2D dmeans, dqs, dls;
} GMMCompressedGrad;
