#include <cblas.h>
#include <math.h>
#include <stdlib.h>

double barr_max(int n, double const *x) {
  int i;
  double m = x[0];
  for (i = 1; i < n; i++) {
    if (m < x[i]) {
      m = x[i];
    }
  }

  return m;
}

double bsqnorm(int n, double const *x) {
  int i;
  double res = x[0] * x[0];
  for (i = 1; i < n; i++) {
    res = res + x[i] * x[i];
  }

  return res;
}

double blog_sum_exp(int n, double const *x) {
  int i;
  double mx = barr_max(n, x);
  double semx = 0.0;

  for (i = 0; i < n; i++) {
    semx = semx + exp(x[i] - mx);
  }

  return log(semx) + mx;
}

__attribute__((const)) double blog_gamma_distrib(double a, double p) {
  int j;
  double out = 0.25 * p * (p - 1) * log(M_PI);

  for (j = 1; j <= p; j++) {
    out = out + lgamma(a + 0.5 * (1 - j));
  }

  return out;
}

double blog_wishart_prior_full_L(int d, int k, double wishart_gamma,
                                 int wishart_m, double *sum_qs, double *Qdiags,
                                 double *Ls) {
  int ik;
  int n = d + wishart_m + 1;
  double C = n * d * (log(wishart_gamma) - 0.5 * log(2)) -
             blog_gamma_distrib(0.5 * n, d);

  double out = 0;
  for (ik = 0; ik < k; ik++) {
    double frobenius =
        bsqnorm(d, &Qdiags[ik * d]) + bsqnorm(d * d, &Ls[ik * d * d]);
    out = out + 0.5 * wishart_gamma * wishart_gamma * (frobenius)-wishart_m *
                    sum_qs[ik];
  }

  return out - k * C;
}

void bpreprocess_qs_full_L(int d, int k, double const *Qs, double *sum_qs,
                           double *Qdiags) {
  int ik, id;
  // int icf_sz = d * (d + 1) / 2;
  for (ik = 0; ik < k; ik++) {
    sum_qs[ik] = 0.;
    for (id = 0; id < d; id++) {
      double q = Qs[ik * d + id];
      sum_qs[ik] = sum_qs[ik] + q;
      Qdiags[ik * d + id] = exp(q);
    }
  }
}

void bQtimesXFullL(int d, double *Qdiag, double *L, double *x, double *out) {
  int i;
  for (i = 0; i < d; i++) {
    out[i] = Qdiag[i] * x[i];
  }
  cblas_dgemv(CblasRowMajor, CblasNoTrans, d, d, 1.0, L, d, x, 1, 1.0, out, 1);
  // cblas_dtrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, d, L, d,
  //             out, 1);

  // for (i = 0; i < d; i++) {
  //   for (j = 0; j < i; j++) {
  //     out[i] += L[i * d + j] * x[j];
  //   }
  // }
}

void blas_gmm_objective(int d, int k, int n, double *alphas, double *means,
                        double *Qs, double *Ls, double *x, double wishart_gamma,
                        int wishart_m, double *err) {
#define int int64_t
  int ix, ik;
  const double CONSTANT = -n * d * 0.5 * log(2 * M_PI);

  double *Qdiags = (double *)malloc(d * k * sizeof(double));
  double *sum_qs = (double *)malloc(k * sizeof(double));
  double *xcentered = (double *)malloc(d * sizeof(double));
  double *Qxcentered = (double *)malloc(d * sizeof(double));
  double *main_term = (double *)malloc(k * sizeof(double));

  bpreprocess_qs_full_L(d, k, Qs, &sum_qs[0], &Qdiags[0]);

  double slse = 0.;
  for (ix = 0; ix < n; ix++) {
    for (ik = 0; ik < k; ik++) {
      // subtract
      for (int i = 0; i < d; i++) {
        xcentered[i] = x[ix * d + i] - means[ik * d + i];
      }
      bQtimesXFullL(d, &Qdiags[ik * d], &Ls[ik * d * d], &xcentered[0],
                    &Qxcentered[0]);
      // two caches for qxcentered at idx 0 and at arbitrary index
      main_term[ik] =
          alphas[ik] + sum_qs[ik] - 0.5 * bsqnorm(d, &Qxcentered[0]);
    }

    // storing cmp for max of main_term
    // 2 x (0 and arbitrary) storing sub to exp
    // storing sum for use in log
    slse = slse + blog_sum_exp(k, &main_term[0]);
  }

  // storing cmp of alphas
  double lse_alphas = blog_sum_exp(k, alphas);

  *err = CONSTANT + slse - n * lse_alphas +
         blog_wishart_prior_full_L(d, k, wishart_gamma, wishart_m, &sum_qs[0],
                                   &Qdiags[0], Ls);

  free(Qdiags);
  free(sum_qs);
  free(xcentered);
  free(Qxcentered);
  free(main_term);
#undef int
}