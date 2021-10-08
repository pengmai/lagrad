#include "ba.h"
#include "mlir_c_abi.h"

extern F64Descriptor1D mlir_compute_reproj_error(
    /*cam=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*X=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*w=*/double *, double *, int64_t, int64_t, int64_t,
    /*feat*/ double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);
extern F64Descriptor1D rodrigues_rotate_point(
    /*rot=*/double *, double *, int64_t, int64_t, int64_t,
    /*X=*/double *, double *, int64_t, int64_t, int64_t);
extern F64Descriptor1D radial_distort(
    /*rad_params=*/double *, double *, int64_t, int64_t, int64_t,
    /*proj=*/double *, double *, int64_t, int64_t, int64_t);
extern F64Descriptor1D project(
    /*cam=*/double *, double *, int64_t, int64_t, int64_t,
    /*X=*/double *, double *, int64_t, int64_t, int64_t);

int main() {
  // const int nCamParams = 11;
  BAInput ba_input = read_ba_data();
  printf("n: %d m: %d p: %d\n", ba_input.n, ba_input.m, ba_input.p);
  // int n = ba_input.n, m = ba_input.m, p = ba_input.p;
  double *deadbeef = (double *)0xdeadbeef;

  double a[11];
  for (int i = 0; i < 11; i++) {
    a[i] = i - 10;
  }

  double b[3] = {4., 5., 6.};
  F64Descriptor1D res = project(deadbeef, a, 0, 11, 1, deadbeef, b, 0, 3, 1);
  printf("project:\n");
  print_d_arr(res.aligned, res.size);
  // F64Descriptor1D reproj_err = mlir_compute_reproj_error(
  //     deadbeef, ba_input.cams, 0, n, nCamParams, nCamParams, 1, deadbeef,
  //     ba_input.X, 0, m, 3, 3, 1, deadbeef, ba_input.w, 0, p, 1, deadbeef,
  //     ba_input.feats, 0, p, 2, 2, 1);

  // printf("Reprojection error:\n");
  // print_d_arr(reproj_err.aligned, reproj_err.size);
}
