#define TARGET_OS_EMBEDDED 0
#include <math.h>
#include <stdlib.h>

#define N {{n}}
#define K {{k}}
#define D {{d}}

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

double log_sum_exp(int n, double const *x) {
  int i;
  double mx = arr_max(n, x);
  double semx = 0.0;

  for (i = 0; i < n; i++) {
    semx = semx + exp(x[i] - mx);
  }

  return log(semx) + mx;
}

double c_main_term(double *alphas, double *means, double *Qs, double *Ls,
                   double *x) {
  double *Qdiags = (double *)malloc(K * D * sizeof(double));
  double *sum_qs = (double *)malloc(K * sizeof(double));
  double *xcentered = (double *)malloc(D * sizeof(double));
  double *Qxcentered = (double *)malloc(D * sizeof(double));
  double *main_term = (double *)malloc(K * sizeof(double));
  // Preprocess Qs
  for (int i = 0; i < K * D; i++) {
    Qdiags[i] = exp(Qs[i]);
  }
  for (int i = 0; i < K; i++) {
    sum_qs[i] = 0;
    for (int j = 0; j < D; j++) {
      sum_qs[i] += Qs[i * D + j];
    }
  }

  double slse = 0;
  for (int ix = 0; ix < N; ix++) {
    for (int ik = 0; ik < K; ik++) {
      // subtract
      for (int j = 0; j < D; j++) {
        xcentered[j] = x[ix * D + j] - means[ik * D + j];
      }

      // Qtimesx
      for (int i = 0; i < D; i++) {
        Qxcentered[i] = Qdiags[ik * D + i] * xcentered[i];
      }
      for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
          Qxcentered[i] += Ls[ik * D * D + i * D + j] * xcentered[j];
        }
      }

      double sqnorm = 0.0;
      for (int i = 0; i < D; i++) {
        sqnorm += Qxcentered[i] * Qxcentered[i];
      }
      main_term[ik] = alphas[ik] + sum_qs[ik] - 0.5 * sqnorm;
    }

    slse += log_sum_exp(K, main_term);
  }

  free(Qdiags);
  free(sum_qs);
  free(xcentered);
  free(Qxcentered);
  free(main_term);

  return slse;
}

extern void __enzyme_autodiff(void *, ...);
extern int enzyme_const;
void enzyme_c_main_term(double *alphas, double *dalphas, double *means,
                        double *dmeans, double *Qs, double *dQs, double *Ls,
                        double *dLs, double *x) {
  __enzyme_autodiff(c_main_term, alphas, dalphas, means, dmeans, Qs, dQs, Ls,
                    dLs, enzyme_const, x);
}
