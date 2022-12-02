#pragma once
#include "lagrad_utils.h"
#include <stdlib.h>
typedef struct TRMVGrad {
  F64Descriptor2D dM;
  F64Descriptor1D dx;
} TRMVGrad;

typedef struct TRMVCompressedGrad {
  F64Descriptor1D dM, dx;
} TRMVCompressedGrad;

TRMVCompressedGrad enzyme_c_trmv_packed(int64_t N, double *M, double *x);

TRMVGrad enzyme_trmv_full(/*M=*/double *, double *, int64_t, int64_t, int64_t,
                          int64_t, int64_t,
                          /*x=*/double *, double *, int64_t, int64_t, int64_t);

TRMVGrad lagrad_trmv_tri_wrapper(int64_t N, double *M, double *x);

TRMVCompressedGrad lagrad_trmv_packed_wrapper(int64_t N, double *M, double *x);

TRMVGrad lagrad_trmv_full_wrapper(int64_t N, double *M, double *x);

TRMVCompressedGrad enzyme_trmv_packed_wrapper(int64_t N, double *M, double *x);

TRMVGrad enzyme_trmv_tri_wrapper(int64_t N, double *M, double *x);

TRMVGrad enzyme_trmv_full_wrapper(int64_t N, double *M, double *x);

TRMVGrad enzyme_trmv_tri(/*M=*/double *, double *, int64_t, int64_t, int64_t,
                         int64_t, int64_t,
                         /*x=*/double *, double *, int64_t, int64_t, int64_t);

TRMVCompressedGrad enzyme_trmv_packed(/*M=*/double *, double *, int64_t,
                                      int64_t, int64_t,
                                      /*x=*/double *, double *, int64_t,
                                      int64_t, int64_t);

TRMVGrad lagrad_trmv_full(/*M=*/double *, double *, int64_t, int64_t, int64_t,
                          int64_t, int64_t,
                          /*x=*/double *, double *, int64_t, int64_t, int64_t);

TRMVGrad lagrad_trmv_tri(/*M=*/double *, double *, int64_t, int64_t, int64_t,
                         int64_t, int64_t,
                         /*x=*/double *, double *, int64_t, int64_t, int64_t);

TRMVCompressedGrad lagrad_trmv_packed(/*M=*/double *, double *, int64_t,
                                      int64_t, int64_t,
                                      /*x=*/double *, double *, int64_t,
                                      int64_t, int64_t);

void expand_ltri(size_t d, double *packed, double *out);

void collapse_ltri(size_t d, double *Ls, double *out);
