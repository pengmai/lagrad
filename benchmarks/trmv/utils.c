#include "trmv.h"

double *deadbeef = (double *)0xdeadbeef;
void expand_ltri(size_t d, double *packed, double *out) {
  size_t Lidx = 0;
  for (size_t j = 0; j < d; j++) {
    for (size_t l = j + 1; l < d; l++) {
      out[l * d + j] = packed[Lidx];
      Lidx++;
    }
  }
}

void collapse_ltri(size_t d, double *Ls, double *out) {
  size_t icf_idx = 0;
  for (size_t j = 0; j < d; j++) {
    for (size_t l = j + 1; l < d; l++) {
      out[icf_idx] = Ls[(l * d) + j];
      icf_idx++;
    }
  }
}

TRMVGrad lagrad_trmv_tri_wrapper(int64_t N, double *M, double *x) {
  return lagrad_trmv_tri(deadbeef, M, 0, N, N, N, 1, deadbeef, x, 0, N, 1);
}

TRMVCompressedGrad lagrad_trmv_packed_wrapper(int64_t N, double *M, double *x) {
  int64_t tri_size = N * (N - 1) / 2;
  return lagrad_trmv_packed(deadbeef, M, 0, tri_size, 1, deadbeef, x, 0, N, 1);
}

TRMVGrad lagrad_trmv_full_wrapper(int64_t N, double *M, double *x) {
  return lagrad_trmv_full(deadbeef, M, 0, N, N, N, 1, deadbeef, x, 0, N, 1);
}

TRMVCompressedGrad enzyme_trmv_packed_wrapper(int64_t N, double *M, double *x) {
  int64_t tri_size = N * (N - 1) / 2;
  return enzyme_trmv_packed(deadbeef, M, 0, tri_size, 1, deadbeef, x, 0, N, 1);
}

TRMVGrad enzyme_trmv_tri_wrapper(int64_t N, double *M, double *x) {
  return enzyme_trmv_tri(deadbeef, M, 0, N, N, N, 1, deadbeef, x, 0, N, 1);
}

TRMVGrad enzyme_trmv_full_wrapper(int64_t N, double *M, double *x) {
  return enzyme_trmv_full(deadbeef, M, 0, N, N, N, 1, deadbeef, x, 0, N, 1);
}