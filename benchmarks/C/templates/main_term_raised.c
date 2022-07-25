#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

extern double sqnorm(int n, double const *x);

void main_term_raised(int d, int k, int n, double const *restrict alphas,
                      double *restrict alphasb, double const *restrict means,
                      double *restrict meansb, double const *restrict Qs,
                      double *restrict Qsb, double const *restrict Ls,
                      double *restrict Lsb, double const *restrict x) {
  int64_t tri_size_conv = d * (d - 1) / 2;
  // int64_t conv2_kd = k * d;
  double *Qdiagsb_ipc21 = calloc(k * d, sizeof(double));
  double *Qdiags_0 = calloc(k * d, sizeof(double));
  double *sum_qs_1 = calloc(k, sizeof(double));
  double *xcentered_2 = calloc(d, sizeof(double));
  // This is %call12
  double *Qxcentered_3 = calloc(d, sizeof(double));
  double *main_term_4 = calloc(k, sizeof(double));

  for (int64_t ik = 0; ik < k; ik++) {
    sum_qs_1[ik] = 0;
    for (int64_t i = 0; i < d; i++) {
      double q = Qs[ik * d + d];
      sum_qs_1[ik] += q;
      Qdiags_0[ik * d + d] = exp(q);
    }
  }

  double *_cache = calloc(n * k * d, sizeof(double));
  double *_cache82 = calloc(n * k, sizeof(double));
  // cache98 holds Qxcentered
  double *_cache98 = calloc(n * k * (d - 1), sizeof(double));
  bool *cmp2_ii_manual_lcssa_cache = calloc(n, sizeof(bool));
  // maual_lcssa126 stores the index of max elements in main_term
  int64_t *manual_lcssa126_cache = calloc(n, sizeof(int64_t));
  double *sub_i135_cache = calloc(n, sizeof(double));
  double slse = 0;
  for (int64_t ix = 0; ix < n; ix++) {
    for (int64_t ik = 0; ik < k; ik++) {
      // Subtract
      for (int64_t i = 0; i < d; i++) {
        xcentered_2[i] = x[ix * d + i] - means[ik * d + i];
      }

      // Qtimesx
      for (int64_t i = 0; i < d; i++) {
        Qxcentered_3[i] = Qdiags_0[ik * d + i] * xcentered_2[i];
      }
      for (int64_t i = 0; i < d; i++) {
        int Lparamsidx = i * (2 * d - i - 1) / 2;
        for (int64_t j = i + 1; j < d; j++) {
          Qxcentered_3[j] +=
              Ls[ik * tri_size_conv + Lparamsidx] * xcentered_2[i];
        }
      }
      memcpy(&_cache98[(ix * k + ik) * (d - 1)], &Qxcentered_3[1], d - 1);
      // end Qtimesx

      main_term_4[ik] =
          alphas[ik] + sum_qs_1[ik] - 0.5 * sqnorm(d, Qxcentered_3);
    }
    // log sum exp
    // arrmax
    double mx = main_term_4[0];
    bool cmp2_ii = false;
    int64_t manual_lcssa126 = 0;
    for (int64_t ik = 0; ik < k; ik++) {
      cmp2_ii = mx < main_term_4[ik];
      if (cmp2_ii) {
        mx = main_term_4[ik];
        manual_lcssa126 = ik;
      }
    }
    // Not sure how this is helpful
    cmp2_ii_manual_lcssa_cache[ix] = cmp2_ii;
    manual_lcssa126_cache[ix] = manual_lcssa126;

    // Again not sure how this helps
    sub_i135_cache[ix] = main_term_4[0] - mx;
    double semx = exp(main_term_4[0] - mx);
    for (int64_t ik = 1; ik < k; ik++) {
      semx += exp(main_term_4[ik] - mx);
    }
    slse += log(semx) + mx;
  }
  free(sum_qs_1);
  // *err = slse;

  /* Beginning of adjoint */

  free(_cache);
  free(_cache82);
  free(_cache98);
  free(Qdiagsb_ipc21);
  free(Qdiags_0);
}
