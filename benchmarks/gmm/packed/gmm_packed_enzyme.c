// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gmm_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

double log_wishart_prior(int p, int k, double wishart_gamma, int wishart_m,
                         double const *sum_qs, double const *Qdiags,
                         double const *icf) {
  int ik;
  int n = p + wishart_m + 1;
  int icf_sz = p * (p + 1) / 2;

  double C = n * p * (log(wishart_gamma) - 0.5 * log(2)) -
             log_gamma_distrib(0.5 * n, p);

  double out = 0;
  for (ik = 0; ik < k; ik++) {
    double frobenius =
        sqnorm(p, &Qdiags[ik * p]) + sqnorm(icf_sz - p, &icf[ik * icf_sz + p]);
    out = out + 0.5 * wishart_gamma * wishart_gamma * (frobenius)-wishart_m *
                    sum_qs[ik];
  }

  // return out - k * C;
  return out;
}

void preprocess_qs(int d, int k, double const *icf, double *sum_qs,
                   double *Qdiags) {
  int ik, id;
  int icf_sz = d * (d + 1) / 2;
  for (ik = 0; ik < k; ik++) {
    sum_qs[ik] = 0.;
    for (id = 0; id < d; id++) {
      double q = icf[ik * icf_sz + id];
      sum_qs[ik] = sum_qs[ik] + q;
      Qdiags[ik * d + id] = exp(q);
    }
  }
}

void Qtimesx(int d, double const *Qdiag,
             double const *ltri, // strictly lower triangular part
             double const *x, double *out) {
  int i, j;
  for (i = 0; i < d; i++) {
    out[i] = Qdiag[i] * x[i];
  }

  // caching lparams as scev doesn't replicate index calculation
  // todo note changing to strengthened form
  for (i = 0; i < d; i++) {
    int Lparamsidx = i * (2 * d - i - 1) / 2;
    for (j = i + 1; j < d; j++) {
      // and this x
      out[j] = out[j] + ltri[Lparamsidx] * x[i];
      Lparamsidx++;
    }
  }
}

void enzyme_gmm_objective(int d, int k, int n, double const *__restrict alphas,
                          double const *__restrict means,
                          double const *__restrict icf,
                          double const *__restrict x, double wishart_gamma,
                          int wishart_m, double *err) {
#define int int64_t
  int ix, ik;
  const double CONSTANT = -n * d * 0.5 * log(2 * M_PI);
  int icf_sz = d * (d + 1) / 2;

  double *Qdiags = (double *)malloc(d * k * sizeof(double));
  double *sum_qs = (double *)malloc(k * sizeof(double));
  double *xcentered = (double *)malloc(d * sizeof(double));
  double *Qxcentered = (double *)malloc(d * sizeof(double));
  double *main_term = (double *)malloc(k * sizeof(double));

  preprocess_qs(d, k, icf, &sum_qs[0], &Qdiags[0]);

  double slse = 0.;
  for (ix = 0; ix < n; ix++) {
    for (ik = 0; ik < k; ik++) {
      subtract(d, &x[ix * d], &means[ik * d], &xcentered[0]);
      Qtimesx(d, &Qdiags[ik * d], &icf[ik * icf_sz + d], &xcentered[0],
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
  *err = slse - n * lse_alphas +
         log_wishart_prior(d, k, wishart_gamma, wishart_m, &sum_qs[0],
                           &Qdiags[0], icf);
  // *err = CONSTANT + slse - n * lse_alphas +
  //        log_wishart_prior(d, k, wishart_gamma, wishart_m, &sum_qs[0],
  //                          &Qdiags[0], icf);

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

// double enzyme_gmm_primal(GMMInput gmm) {
//   double err = 0.0;
//   enzyme_gmm_objective(gmm.d, gmm.k, gmm.n, gmm.alphas, gmm.means, gmm.icf,
//                        gmm.x, gmm.wishart_gamma, gmm.wishart_m, &err);
//   return err;
// }

GMMCompressedGrad enzyme_c_gmm_packed(GMMInput *gmm, double *compressed_Ls) {
  int d = gmm->d, k = gmm->k, n = gmm->n;
  int tri_size = d * (d - 1) / 2;
  int icf_size = d * (d + 1) / 2;
  double *dalphas_buf = calloc(k, sizeof(double));
  double *dmeans_buf = calloc(k * d, sizeof(double));
  double *dQs_buf = calloc(k * d, sizeof(double));
  double *dLs_buf = calloc(k * tri_size, sizeof(double));

  F64Descriptor1D dalphas = {.allocated = dalphas_buf,
                             .aligned = dalphas_buf,
                             .offset = 0,
                             .size = k,
                             .stride = 1};

  F64Descriptor2D dmeans = {.allocated = dmeans_buf,
                            .aligned = dmeans_buf,
                            .offset = 0,
                            .size_0 = k,
                            .size_1 = d,
                            .stride_0 = d,
                            .stride_1 = 1};
  double *icfb = calloc(k * (tri_size + d), sizeof(double));
  F64Descriptor2D dQs = {.allocated = dQs_buf,
                         .aligned = dQs_buf,
                         .offset = 0,
                         .size_0 = k,
                         .size_1 = d,
                         .stride_0 = d,
                         .stride_1 = 1};
  F64Descriptor2D dLs = {.allocated = dLs_buf,
                         .aligned = dLs_buf,
                         .offset = 0,
                         .size_0 = k,
                         .size_1 = tri_size,
                         .stride_0 = tri_size,
                         .stride_1 = 1};

  double err = 0.0, errb = 1.0;
  __enzyme_autodiff(enzyme_gmm_objective, enzyme_const, gmm->d, enzyme_const,
                    gmm->k, enzyme_const, gmm->n, enzyme_dup, gmm->alphas,
                    dalphas.aligned, enzyme_dup, gmm->means, dmeans.aligned,
                    enzyme_dup, gmm->icf, icfb, enzyme_const, gmm->x,
                    enzyme_const, gmm->wishart_gamma, enzyme_const,
                    gmm->wishart_m, enzyme_dupnoneed, &err, &errb);
  for (int ik = 0; ik < k; ik++) {
    memcpy(&dQs.aligned[ik * d], &icfb[ik * icf_size], d * sizeof(double));
    memcpy(&dLs.aligned[ik * tri_size], &icfb[ik * icf_size + d],
           tri_size * sizeof(double));
  }
  free(icfb);
  GMMCompressedGrad grad = {
      .dalphas = dalphas, dmeans = dmeans, .dqs = dQs, .dls = dLs};
  return grad;
}
