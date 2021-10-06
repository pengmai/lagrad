#include "ba.h"
#include "mlir_c_abi.h"

extern F64Descriptor1D mlir_compute_reproj_error(
    /*cam=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*X=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*w=*/double *, double *, int64_t, int64_t, int64_t,
    /*feat*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);

int main() {
  const int nCamParams = 11;
  BAInput ba_input = read_ba_data();
  printf("n: %d m: %d p: %d\n", ba_input.n, ba_input.m, ba_input.p);
  int n = ba_input.n, m = ba_input.m, p = ba_input.p;
  double *deadbeef = (double *)0xdeadbeef;
  F64Descriptor1D reproj_err = mlir_compute_reproj_error(
      deadbeef, ba_input.cams, 0, n, nCamParams, nCamParams, 1, deadbeef,
      ba_input.X, 0, m, 3, 3, 1, deadbeef, ba_input.w, 0, p, 1, deadbeef,
      ba_input.feats, 0, p, 2, 2, 1);

  printf("Reprojection error:\n");
  print_d_arr(reproj_err.aligned, reproj_err.size);
}
