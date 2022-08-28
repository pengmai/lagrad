// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gmm_types.h"
#include "lagrad_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ==================================================================== */
/*                                UTILS                                 */
/* ==================================================================== */

// This throws error on n<1
double arr_max(int n, double const *x) {
  int i;
  double m = x[0];
  for (i = 1; i < n; i++) {
    if (m < x[i]) {
      m = x[i];
    }
  }

  return m;
}

// sum of component squares
double sqnorm(int n, double const *x) {
  int i;
  double res = x[0] * x[0];
  for (i = 1; i < n; i++) {
    res = res + x[i] * x[i];
  }

  return res;
}

// out = a - b
void subtract(int d, double const *x, double const *y, double *out) {
  int id;
  for (id = 0; id < d; id++) {
    out[id] = x[id] - y[id];
  }
}

double log_sum_exp(int n, double const *x) {
  int i;
  double mx = arr_max(n, x);
  double semx = 0.0;

  for (i = 0; i < n; i++) {
    semx = semx + exp(x[i] - mx);
  }

  return log(semx) + mx;
}

__attribute__((const)) double log_gamma_distrib(double a, double p) {
  int j;
  double out = 0.25 * p * (p - 1) * log(M_PI);

  for (j = 1; j <= p; j++) {
    out = out + lgamma(a + 0.5 * (1 - j));
  }

  return out;
}

/* ======================================================================== */
/*                                MAIN LOGIC                                */
/* ======================================================================== */

double log_wishart_prior_full_L(int d, int k, double wishart_gamma,
                                int wishart_m, double *sum_qs, double *Qdiags,
                                double *Ls) {
  int ik;
  int n = d + wishart_m + 1;
  double C = n * d * (log(wishart_gamma) - 0.5 * log(2)) -
             log_gamma_distrib(0.5 * n, d);

  double out = 0;
  for (ik = 0; ik < k; ik++) {
    double frobenius =
        sqnorm(d, &Qdiags[ik * d]) + sqnorm(d * d, &Ls[ik * d * d]);
    out = out + 0.5 * wishart_gamma * wishart_gamma * (frobenius)-wishart_m *
                    sum_qs[ik];
  }

  return out - k * C;
}

void preprocess_qs_full_L(int d, int k, double const *Qs, double *sum_qs,
                          double *Qdiags) {
  int ik, id;
  int icf_sz = d * (d + 1) / 2;
  for (ik = 0; ik < k; ik++) {
    sum_qs[ik] = 0.;
    for (id = 0; id < d; id++) {
      double q = Qs[ik * d + id];
      sum_qs[ik] = sum_qs[ik] + q;
      Qdiags[ik * d + id] = exp(q);
    }
  }
}

void QtimesXFullL(int d, double *Qdiag, double *L, double *x, double *out) {
  int i, j;
  for (i = 0; i < d; i++) {
    out[i] = Qdiag[i] * x[i];
  }

  for (i = 0; i < d; i++) {
    for (j = 0; j < d; j++) {
      out[i] += L[i * d + j] * x[j];
    }
  }
}

// Qs: tensor<k x d>
// Ls: tensor<k x d x d> (lower triangular)
void enzyme_gmm_objective_full_L(int d, int k, int n, double *alphas,
                                 double *means, double *Qs, double *Ls,
                                 double *x, double wishart_gamma, int wishart_m,
                                 double *err) {
#define int int64_t
  int ix, ik;
  const double CONSTANT = -n * d * 0.5 * log(2 * M_PI);

  double *Qdiags = (double *)malloc(d * k * sizeof(double));
  double *sum_qs = (double *)malloc(k * sizeof(double));
  double *xcentered = (double *)malloc(d * sizeof(double));
  double *Qxcentered = (double *)malloc(d * sizeof(double));
  double *main_term = (double *)malloc(k * sizeof(double));

  preprocess_qs_full_L(d, k, Qs, &sum_qs[0], &Qdiags[0]);

  double slse = 0.;
  for (ix = 0; ix < n; ix++) {
    for (ik = 0; ik < k; ik++) {
      subtract(d, &x[ix * d], &means[ik * d], &xcentered[0]);
      QtimesXFullL(d, &Qdiags[ik * d], &Ls[ik * d * d], &xcentered[0],
                   &Qxcentered[0]);
      // if (ix == n - 1 && ik == 0) {
      //   __print_d_arr(&Qxcentered[0], d);
      // }
      // two caches for qxcentered at idx 0 and at arbitrary index
      main_term[ik] = alphas[ik] + sum_qs[ik] - 0.5 * sqnorm(d, &Qxcentered[0]);
    }

    // storing cmp for max of main_term
    // 2 x (0 and arbitrary) storing sub to exp
    // storing sum for use in log
    slse = slse + log_sum_exp(k, &main_term[0]);
  }

  // storing cmp of alphas
  double lse_alphas = log_sum_exp(k, alphas);

  *err = CONSTANT + slse - n * lse_alphas +
         log_wishart_prior_full_L(d, k, wishart_gamma, wishart_m, &sum_qs[0],
                                  &Qdiags[0], Ls);

  free(Qdiags);
  free(sum_qs);
  free(xcentered);
  free(Qxcentered);
  free(main_term);
#undef int
}

extern int enzyme_const;
extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern void __enzyme_autodiff(void *, ...);

double enzyme_gmm_primal_full(GMMInput gmm) {
  double err = 0.0;
  enzyme_gmm_objective_full_L(gmm.d, gmm.k, gmm.n, gmm.alphas, gmm.means,
                              gmm.Qs, gmm.Ls, gmm.x, gmm.wishart_gamma,
                              gmm.wishart_m, &err);
  return err;
}

GMMGrad enzyme_c_gmm_full(GMMInput *gmm) {
  int d = gmm->d, k = gmm->k, n = gmm->n;

  F64Descriptor1D dalphas = {.allocated = NULL,
                             .aligned = calloc(k, sizeof(double)),
                             .offset = 0,
                             .size = k,
                             .stride = 1};

  F64Descriptor2D dmeans = {.allocated = NULL,
                            .aligned = calloc(k * d, sizeof(double)),
                            .offset = 0,
                            .size_0 = k,
                            .size_1 = d,
                            .stride_0 = d,
                            .stride_1 = 1};
  F64Descriptor2D dQs = {.allocated = NULL,
                         .aligned = calloc(k * d, sizeof(double)),
                         .offset = 0,
                         .size_0 = k,
                         .size_1 = d,
                         .stride_0 = d,
                         .stride_1 = 1};
  F64Descriptor3D dLs = {.allocated = NULL,
                         .aligned = calloc(k * d * d, sizeof(double)),
                         .offset = 0,
                         .size_0 = k,
                         .size_1 = d,
                         .size_2 = d,
                         .stride_0 = d * d,
                         .stride_1 = d,
                         .stride_2 = 1};

  double err = 0.0, errb = 1.0;
  __enzyme_autodiff(enzyme_gmm_objective_full_L, enzyme_const, gmm->d,
                    enzyme_const, gmm->k, enzyme_const, gmm->n, enzyme_dup,
                    gmm->alphas, dalphas.aligned, enzyme_dup, gmm->means,
                    dmeans.aligned, enzyme_dup, gmm->Qs, dQs.aligned,
                    enzyme_dup, gmm->Ls, dLs.aligned, enzyme_const, gmm->x,
                    enzyme_const, gmm->wishart_gamma, enzyme_const,
                    gmm->wishart_m, enzyme_dupnoneed, &err, &errb);
  GMMGrad grad = {.dalphas = dalphas, dmeans = dmeans, .dqs = dQs, .dls = dLs};
  return grad;
}
