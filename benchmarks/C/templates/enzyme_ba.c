#define TARGET_OS_EMBEDDED 0
#include <math.h>
#include <stdlib.h>

double sqsum(int n, double const *x) {
  int i;
  double res = 0;
  for (i = 0; i < n; i++) {
    res = res + x[i] * x[i];
  }

  return res;
}

double ecross(double const *a, double const *b, double *out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

void erodrigues_rotate_point(double const *__restrict rot,
                             double const *__restrict pt,
                             double *__restrict rotatedPt) {
  int i;
  double sqtheta = sqsum(3, rot);
  if (sqtheta != 0) {
    double theta, costheta, sintheta, theta_inverse;
    double w[3], w_cross_pt[3], tmp;

    theta = sqrt(sqtheta);
    costheta = cos(theta);
    sintheta = sin(theta);
    theta_inverse = 1.0 / theta;

    for (i = 0; i < 3; i++) {
      w[i] = rot[i] * theta_inverse;
      rotatedPt[i] = w[i];
    }

    // ecross(w, pt, w_cross_pt);
    // ecross(w, pt, rotatedPt);

    // tmp = (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (1. - costheta);

    // for (i = 0; i < 3; i++) {
    //   rotatedPt[i] = pt[i] * costheta + w_cross_pt[i] * sintheta + w[i] *
    //   tmp;
    // }
  } else {
    double rot_cross_pt[3];
    ecross(rot, pt, rot_cross_pt);

    for (i = 0; i < 3; i++) {
      rotatedPt[i] = pt[i] + rot_cross_pt[i];
    }
  }
}

void eradial_distort(double const *rad_params, double *proj) {
  double rsq, L;
  rsq = sqsum(2, proj);
  L = 1. + rad_params[0] * rsq + rad_params[1] * rsq * rsq;
  proj[0] = proj[0] * L;
  proj[1] = proj[1] * L;
}

void eproject(double const *__restrict cam, double const *__restrict X,
              double *__restrict proj) {
  double const *C = &cam[3];
  double Xo[3], Xcam[3];

  Xo[0] = X[0] - C[0];
  Xo[1] = X[1] - C[1];
  Xo[2] = X[2] - C[2];

  erodrigues_rotate_point(&cam[0], Xo, Xcam);
  proj[0] = Xcam[0];
  proj[1] = Xcam[1];

  // proj[0] = Xcam[0] / Xcam[2];
  // proj[1] = Xcam[1] / Xcam[2];

  // eradial_distort(&cam[9], proj);

  // proj[0] = proj[0] * cam[6] + cam[7];
  // proj[1] = proj[1] * cam[6] + cam[8];
}

extern int enzyme_const;
extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern void __enzyme_autodiff(void *, ...);

void ecompute_reproj_error(double const *__restrict cam,
                           double const *__restrict X,
                           double const *__restrict w,
                           double const *__restrict feat,
                           double *__restrict err) {
  double proj[2];
  eproject(cam, X, proj);

  err[0] = proj[0];
  err[1] = proj[1];
  // err[0] = (*w) * (proj[0] - feat[0]);
  // err[1] = (*w) * (proj[1] - feat[1]);
}

void ecompute_zach_weight_error(double const *w, double *err) {
  *err = 1 - (*w) * (*w);
}

void enzyme_c_compute_reproj_error(double const *cam, double *dcam,
                                   double const *X, double *dX, double const *w,
                                   double *wb, double const *feat, double *err,
                                   double *derr) {
  __enzyme_autodiff(ecompute_reproj_error, enzyme_dup, cam, dcam, enzyme_dup, X,
                    dX, enzyme_dup, w, wb, enzyme_const, feat, enzyme_dupnoneed,
                    err, derr);
}

void enzyme_c_compute_w_error(double const *w, double *wb, double *err,
                              double *derr) {
  __enzyme_autodiff(ecompute_zach_weight_error, enzyme_dup, w, wb,
                    enzyme_dupnoneed, err, derr);
}
