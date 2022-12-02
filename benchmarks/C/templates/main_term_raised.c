#define TARGET_OS_EMBEDDED 0
#include <math.h>
#include <mlir_c_abi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern double sqnorm(int n, double const *x);
// double sqnorm(int n, double const *x) {
//   int i;
//   double res = x[0] * x[0];
//   for (i = 1; i < n; i++) {
//     res = res + x[i] * x[i];
//   }

//   return res;
// }

void main_term_raised(int d, int k, int n, double const *restrict alphas,
                      double *restrict alphasb, double const *restrict means,
                      double *restrict meansb, double const *restrict Qs,
                      double *restrict Qsb, double const *restrict Ls,
                      double *restrict Lsb, double const *restrict x) {
  int64_t tri_size_conv = d * (d - 1) / 2;
  // int64_t conv2_kd = k * d;
  double *Qdiags_0 = calloc(k * d, sizeof(double));
  double *Qdiagsb_ipc21 = calloc(k * d, sizeof(double));
  double *sum_qs_1 = calloc(k, sizeof(double));
  double *sum_qsb_ipc = calloc(k, sizeof(double));
  double *xcentered_2 = calloc(d, sizeof(double));
  double *xcenteredb_ipc37 = calloc(d, sizeof(double));
  // This is %call12
  double *Qxcentered_3 = calloc(d, sizeof(double));
  double *Qxcenteredb_ipc40 = calloc(d, sizeof(double));
  double *main_term_4 = calloc(k, sizeof(double));
  double *main_termb_ipc116 = calloc(k, sizeof(double));

  for (int64_t ik = 0; ik < k; ik++) {
    sum_qs_1[ik] = 0;
    for (int64_t i = 0; i < d; i++) {
      double q = Qs[ik * d + i];
      sum_qs_1[ik] += q;
      Qdiags_0[ik * d + i] = exp(q);
    }
  }

  // cache holds xcentered
  double *_cache = calloc(n * k * d, sizeof(double));
  double *_cache82 = calloc(n * k, sizeof(double));
  // cache98 holds Qxcentered
  double *_cache98 = calloc(n * k * (d - 1), sizeof(double));
  bool *cmp2_ii_manual_lcssa_cache = calloc(n, sizeof(bool));
  // maual_lcssa126 stores the index of max elements in main_term
  int64_t *manual_lcssa126_cache = calloc(n, sizeof(int64_t));
  double *sub_i135_cache = calloc(n, sizeof(double));
  double *sub_i_cache = calloc(n * (k - 1), sizeof(double));
  double *add_imanual_lcssa154_cache = calloc(n, sizeof(double));
  double slse = 0;
  for (int64_t ix = 0; ix < n; ix++) {
    for (int64_t ik = 0; ik < k; ik++) {
      // Subtract
      for (int64_t i = 0; i < d; i++) {
        xcentered_2[i] = x[ix * d + i] - means[ik * d + i];
      }
      memcpy(&_cache[ix * k * d + ik * d], xcentered_2, d * sizeof(double));

      // Qtimesx
      for (int64_t i = 0; i < d; i++) {
        Qxcentered_3[i] = Qdiags_0[ik * d + i] * xcentered_2[i];
      }
      for (int64_t i = 0; i < d; i++) {
        int64_t Lparamsidx = i * (2 * d - i - 1) / 2;
        for (int64_t j = i + 1; j < d; j++) {
          Qxcentered_3[j] +=
              Ls[ik * tri_size_conv + Lparamsidx] * xcentered_2[i];
          Lparamsidx++;
        }
      }
      _cache82[ix * k + ik] = Qxcentered_3[0];
      memcpy(&_cache98[(ix * k + ik) * (d - 1)], &Qxcentered_3[1],
             (d - 1) * sizeof(double));
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
    double sub_i = main_term_4[0] - mx;
    sub_i135_cache[ix] = sub_i;
    double semx = exp(sub_i);
    for (int64_t ik = 1; ik < k; ik++) {
      sub_i = main_term_4[ik] - mx;
      sub_i_cache[ix * (k - 1) + ik - 1] = sub_i;
      // printf("subi : %.4e at idx %lld\n", sub_i, ix * (k - 1) + ik - 1);
      semx += exp(sub_i);
    }
    add_imanual_lcssa154_cache[ix] = semx;

    // printf("mainb: %.4e\n", add_imanual_lcssa154_cache[ix]);
    slse += log(semx) + mx;
  }
  free(sum_qs_1);
  // This needed to be manually inserted
  free(xcentered_2);
  free(Qxcentered_3);
  free(main_term_4);
  // end of manual insertion
  // *err = slse;

  /* Beginning of adjoint */
  double add8_i_de = 0;
  double add47_de = 1.0;
  double add_i136_de = 0.0;
  double add_i138_de = 0.0;
  // DON'T GET ADD_I AND ADD_1_I MIXED UP
  double add_de = 0;
  double add_i_de = 0;
  double add_1_i_de = 0;
  double de26 = 0;
  double de30 = 0;
  double de31 = 0;
  double de32 = 0;
  double de33 = 0;
  double de34 = 0;
  double de_131 = 0;
  double de_133 = 0;
  double de_139 = 0;
  double de_150 = 0;
  double sub_i_de = 0;
  double sub43_de = 0;
  double sub_i135_de = 0;
  double m_0_lcssa_ii_de = 0;
  double m_1_ii_de = 0;
  double pre_146_de = 0;
  double slse_143_de = 0;
  double semx_0_lcssa_i_de = 0;
  for (int64_t ix = n - 1; ix >= 0; ix--) {
    // invertlog_sum_exp.exit
    double _636 = add47_de;
    add47_de = 0;
    slse_143_de += _636;
    add_1_i_de += _636;
    double _641 = add_1_i_de;
    add_1_i_de = 0;
    m_0_lcssa_ii_de += _641;
    de_150 += _641;
    double _646 = de_150;
    de_150 = 0;

    // invertlog_sum_exp.exit_phimerge
    semx_0_lcssa_i_de += _646 / add_imanual_lcssa154_cache[ix]; // %660
    double _663 = semx_0_lcssa_i_de;

    // printf("663: %.4e\n", _663);
    semx_0_lcssa_i_de = 0;

    add_i_de += _663;
    add_i136_de += 0.0;

    // %659 will take this value if k == 1, which is always false.
    // double add_i136_unwrap = exp(sub_i135_cache[ix]) + 0.0;

    for (int64_t ik = k - 2; ik >= 0; ik--) {
      /* invertlog_sum_exp.exit.loopexit */
      /* mergeinvertfor.body.for.body_crit_edge.i_log_sum_exp.exit.loopexit
       * (initializes and assigns ik) */
      /* invertfor.body.for.body_crit_edge.i */
      double _587 = add_i_de;

      add_i_de = 0;
      add_i138_de += _587;
      de_139 += _587;
      double _592 = de_139;
      de_139 = 0;
      // DEBUG_POINT_0

      sub_i_de = _592 * exp(sub_i_cache[ix * (k - 1) + ik]);
      // printf("mainb: %.4e\n", _587);
      m_0_lcssa_ii_de += (-sub_i_de); // %611

      main_termb_ipc116[ik + 1] += sub_i_de;
      double _621 = add_i138_de;
      add_i138_de = 0;

      add_i136_de = (ik == 0) ? (add_i136_de + _621) : add_i136_de;
      add_i_de = (ik == 0) ? add_i_de : (add_i_de + _621);
      // DEBUG_POINT_1
    }
    /* invertfor.body.for.body_crit_edge.i.preheader */
    /* invertfor.body.preheader.i */
    // DEBUG_POINT_2
    de_133 += add_i136_de;
    add_i136_de = 0;
    sub_i135_de += de_133 * exp(sub_i135_cache[ix]);
    de_133 = 0;
    de31 += sub_i135_de;
    m_0_lcssa_ii_de += (-sub_i135_de);
    sub_i135_de = 0;

    // printf("mainb: %.4e\n", m_0_lcssa_ii_de);
    /* invertarr_max.exit.i */
    m_0_lcssa_ii_de = 0;
    m_1_ii_de += m_0_lcssa_ii_de;
    de31 += 0;
    /* invertarr_max.exit.i.loopexit */
    int64_t _unwrap130 =
        cmp2_ii_manual_lcssa_cache[ix] ? k - 1 : manual_lcssa126_cache[ix];

    for (int64_t ik = k - 2; ik >= 0; ik--) {
      /* mergeinvertfor.body.i.i_arr_max.exit.i.loopexit */
      de_131 += (_unwrap130 == ik + 1) ? de_131 + m_1_ii_de : de_131;
      main_termb_ipc116[ik + 1] += de_131;
      de_131 = 0;
      de31 = (ik == 0) ? de31 + (_unwrap130 == 0 ? m_1_ii_de : 0.0) : de31;
      m_1_ii_de = (ik == 0) ? 0.0 : m_1_ii_de;
    }
    /* invertfor.body.i.i.preheader */
    /* invertfor.end */

    // DEBUG_POINT_3
    pre_146_de += de31;
    de31 = 0;

    /* invertfor.end.loopexit */
    main_termb_ipc116[0] += pre_146_de;
    pre_146_de = 0;
    /* mergeinvertfor.body23_for.end.loopexit */
    for (int64_t ik = k - 1; ik >= 0; ik--) {
      /* invertsqnorm.exit */
      // DEBUG_POINT_4
      sub43_de += main_termb_ipc116[ik];
      main_termb_ipc116[ik] = 0;
      add_de += sub43_de;
      double add_i106_de = 0;
      add_i106_de += -sub43_de * 0.5;
      sub43_de = 0;
      double mul_i102_de = 0;

      /* invertsqnorm.exit.loopexit */
      for (int64_t i = d - 2; i >= 0; i--) {
        /* mergeinvertfor.body.i109_sqnorm.exit.loopexit */
        /* invertfor.body.i109 */
        // DEBUG_POINT_5
        Qxcenteredb_ipc40[i + 1] =
            2 * add_i106_de * _cache98[ix * k * (d - 1) + ik * (d - 1) + i];
        double _451 = add_i106_de;
        mul_i102_de += (i == 0) ? add_i106_de : 0.0;
        add_i106_de = 0;
        add_i106_de += (i == 0) ? 0 : _451;
      }
      /* invertfor.body.i109.preheader */
      /* invertcQtimesx.exit */
      // DEBUG_POINT_6
      // looks correct
      double m0diffe91 = mul_i102_de * _cache82[ix * k + ik];
      mul_i102_de = 0;
      double _385 = add_de;
      add_de = 0;
      double de93 = 0;
      de93 += _385;
      double de94 = 0;
      de94 += _385;
      double _390 = de94;
      de94 = 0;
      // DEBUG_POINT_7
      sum_qsb_ipc[ik] += _390;
      double _395 = de93;
      de93 = 0;
      alphasb[ik] += _395;

      /* invertcQtimesx.exit.loopexit */
      double _362 = m0diffe91 + m0diffe91; // this matches
      Qxcenteredb_ipc40[0] = _362;
      /* mergeinvertfor.body7.i_cQtimesx.exit.loopexit */
      for (int64_t i = d - 1; i >= 0; i--) {
        /* invertfor.cond5.loopexit.i */
        if (i + 1 < d) {
          /* invertfor.cond5.loopexit.i.loopexit */
          /* mergeinvertfor.body13.i_for.cond5.loopexit.i.loopexit */
          for (int64_t j = d - 2 - i; j >= 0; j--) {
            /* invertfor.body13.i */
            // DEBUG_POINT_8
            int64_t Lidx = (2 * d + (((int32_t)i) ^ -1)) * i / 2 + j;
            xcenteredb_ipc37[i] +=
                Qxcenteredb_ipc40[i + 1 + j] * Ls[ik * tri_size_conv + Lidx];
            Lsb[ik * tri_size_conv + Lidx] +=
                Qxcenteredb_ipc40[i + 1 + j] * _cache[ix * k * d + ik * d + i];
          }
          /* invertfor.body13.lr.ph.i */
          /* invertfor.body7.i */
        }
      }
      /* invertfor.body7.i.preheader */
      /* mergeinvertfor.body.i114_for.body7.i.preheader */
      for (int64_t i = d - 1; i >= 0; i--) {
        /* invertfor.body.i114 */
        // DEBUG_POINT_9
        xcenteredb_ipc37[i] += Qxcenteredb_ipc40[i] * Qdiags_0[ik * d + i];
        Qdiagsb_ipc21[ik * d + i] +=
            Qxcenteredb_ipc40[i] * _cache[ix * k * d + ik * d + i];
      }
      /* invertfor.body.i114.preheader */
      // DEBUG_POINT_10
      for (int64_t i = d - 1; i >= 0; i--) {
        /* invertfor.body.i128 */
        meansb[ik * d + i] += -xcenteredb_ipc37[i];
        xcenteredb_ipc37[i] = 0;
      }
      /* invertfor.body.i128.preheader */
      /* invertfor.body23 */
      double _191 = de34;
      de34 = 0;
      de32 += (ik == 0) ? _191 : 0.0;
    }
    /* invertfor.body23.lr.ph */
    /* invertfor.cond19.preheader */
    // DEBUG_POINT_11
    de31 += (ix == 0) ? 0.0 : de30;
    de30 = 0;
    double _179 = de32;
    de32 = 0;
    de33 += (ix == 0) ? 0.0 : _179;
    double _184 = slse_143_de;
    slse_143_de = 0;
    add47_de += (ix == 0) ? 0.0 : _184;
  }
  /* invertfor.cond19.preheader.lr.ph */
  free(_cache);
  free(_cache82);
  free(_cache98);
  free(cmp2_ii_manual_lcssa_cache);
  free(manual_lcssa126_cache);
  free(sub_i135_cache);
  free(sub_i_cache);
  free(add_imanual_lcssa154_cache);

  /* invertpreprocess_qs.exit */
  /* invertpreprocess_qs.exit.loopexit */
  for (int64_t ik = k - 1; ik >= 0; ik--) {
    /* invertfor.inc15.i */
    // DEBUG_POINT_12
    /* invertfor.inc15.i.loopexit */
    add8_i_de += sum_qsb_ipc[ik];
    for (int64_t i = d - 1; i >= 0; i--) {
      /* invertfor.body3.i */
      double _125 = add8_i_de;
      de26 += _125;
      // double _130 = add8_i_de + Qdiagsb_ipc21[ik * d + i] * exp(Qs[ik * d +
      // i]);
      Qsb[ik * d + i] +=
          add8_i_de + Qdiagsb_ipc21[ik * d + i] * exp(Qs[ik * d + i]);
      add8_i_de = 0;
      Qdiagsb_ipc21[ik * d + i] = 0;
      add8_i_de += (i == 0) ? 0.0 : de26;
      de26 = 0;
    }
    /* invertfor.body3.lr.ph.i */
    /* invertfor.body.i */
    sum_qsb_ipc[ik] = 0;
  }
  /* invertfor.body.lr.ph.i */
  /* invertentry */
  free(main_termb_ipc116);
  free(Qxcenteredb_ipc40);
  free(xcenteredb_ipc37);
  free(sum_qsb_ipc);
  free(Qdiagsb_ipc21);
  free(Qdiags_0);
}
