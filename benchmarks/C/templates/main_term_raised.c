#include <math.h>
#include <mlir_c_abi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// extern double sqnorm(int n, double const *x);
double sqnorm(int n, double const *x) {
  int i;
  double res = x[0] * x[0];
  for (i = 1; i < n; i++) {
    res = res + x[i] * x[i];
  }

  return res;
}

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
        int Lparamsidx = i * (2 * d - i - 1) / 2;
        for (int64_t j = i + 1; j < d; j++) {
          Qxcentered_3[j] +=
              Ls[ik * tri_size_conv + Lparamsidx] * xcentered_2[i];
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
  // *err = slse;

  /* Beginning of adjoint */
  double add21_i_de = 0;
  double add47_de = 1.0;
  double add_i136_de = 0.0;
  double add_i138_de = 0.0;
  // DON'T GET ADD_I AND ADD_1_I MIXED UP
  double add_de = 0;
  double add_i_de = 0;
  double add_1_i_de = 0;
  double mul20_i_de = 0;
  double mul42_de = 0;
  double mul_i113_de = 0;
  double de30 = 0;
  double de31 = 0;
  double de32 = 0;
  double de33 = 0;
  double de34 = 0;
  double de35 = 0;
  double de38 = 0;
  double de50 = 0;
  double de51 = 0;
  double de59 = 0;
  double de62 = 0;
  double de79 = 0;
  double de_131 = 0;
  double de_133 = 0;
  double de_139 = 0;
  double de_150 = 0;
  double sub_i_de = 0;
  double sub43_de = 0;
  double sub_i135_de = 0;
  double sub_i124_de = 0;
  double m_0_lcssa_ii_de = 0;
  double m_1_ii_de = 0;
  double pre_de = 0;
  double pre_i_101_de = 0;
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

      sub_i_de +=
          _592 *
          exp(sub_i_cache[ix * (k - 1) + ik]); // sub_i_cache looks correct
      // printf("mainb: %.4e\n", _587);
      double _610 = sub_i_de;
      sub_i_de = 0;
      pre_i_101_de += _610;
      m_0_lcssa_ii_de += (-_610); // %611
      double _616 = pre_i_101_de;
      pre_i_101_de = 0;

      main_termb_ipc116[ik + 1] += _616;
      double _621 = add_i138_de;
      add_i138_de = 0;

      add_i136_de = (ik == 0) ? (add_i136_de + _621) : add_i136_de;
      add_i_de = (ik == 0) ? add_i_de : (add_i_de + _621);
      // DEBUG_POINT_1
    }
    /* invertfor.body.for.body_crit_edge.i.preheader */
    /* invertfor.body.preheader.i */
    // DEBUG_POINT_2
    double _567 = add_i136_de;
    add_i136_de = 0;
    de_133 += _567;
    double _570 = de_133;
    de_133 = 0;
    sub_i135_de += _570 * exp(sub_i135_cache[ix]);
    double _581 = sub_i135_de;
    sub_i135_de = 0;
    de31 += _581;
    m_0_lcssa_ii_de += (-_581);

    // printf("mainb: %.4e\n", m_0_lcssa_ii_de);
    /* invertarr_max.exit.i */
    double _556 = m_0_lcssa_ii_de;
    m_0_lcssa_ii_de = 0;
    m_1_ii_de += _556;
    de31 += 0;
    /* invertarr_max.exit.i.loopexit */
    int64_t _unwrap130 =
        cmp2_ii_manual_lcssa_cache[ix] ? k - 1 : manual_lcssa126_cache[ix];

    for (int64_t ik = k - 2; ik >= 0; ik--) {
      /* mergeinvertfor.body.i.i_arr_max.exit.i.loopexit */
      de_131 += (_unwrap130 == ik + 1) ? de_131 + m_1_ii_de : de_131;
      double _524 = de_131;
      de_131 = 0;
      main_termb_ipc116[ik + 1] += _524;
      double _533 = m_1_ii_de;
      m_1_ii_de = (ik == 0) ? 0.0 : m_1_ii_de;
      de31 = (ik == 0) ? de31 + (_unwrap130 == 0 ? _533 : 0.0) : de31;
      // printf("de_31 : %.4e\n", de_31);
    }
    /* invertfor.body.i.i.preheader */
    /* invertfor.end */

    // DEBUG_POINT_3
    double _497 = de31;
    de31 = 0;
    pre_146_de += _497;
    de30 += 0;

    double _507 = de33;
    de33 += 0;
    de35 += _507;
    de32 += 0;
    /* invertfor.end.loopexit */
    double _493 = pre_146_de;
    pre_146_de = 0;
    main_termb_ipc116[0] += _493;
    /* mergeinvertfor.body23_for.end.loopexit */
    for (int64_t ik = k - 1; ik >= 0; ik--) {
      /* invertsqnorm.exit */
      // DEBUG_POINT_4
      double _469 = main_termb_ipc116[ik];
      main_termb_ipc116[ik] = 0;
      sub43_de += _469;
      double _472 = sub43_de;
      double _473 = -sub43_de;
      sub43_de = 0;

      add_de += _472;
      mul42_de += _473;

      double m0differes = mul42_de * 0.5;
      mul42_de = 0;

      double res_0_lcssa_i_de = 0;
      res_0_lcssa_i_de += m0differes;
      double _481 = res_0_lcssa_i_de;
      res_0_lcssa_i_de = 0;

      double add_i106_de = 0;
      add_i106_de += _481;
      double mul_i102_de = 0;
      mul_i102_de += 0;

      /* invertsqnorm.exit.loopexit */
      for (int64_t i = d - 2; i >= 0; i--) {
        /* mergeinvertfor.body.i109_sqnorm.exit.loopexit */
        /* invertfor.body.i109 */
        // DEBUG_POINT_5
        double _412 = add_i106_de;
        add_i106_de = 0;
        double res_17_i_de = 0;
        res_17_i_de += _412;
        double mul5_i_de = 0;
        mul5_i_de += _412;
        double _440 = _cache98[ix * k * (d - 1) + ik * (d - 1) + i];
        double m0diffe110 = mul5_i_de * _440;
        mul5_i_de = 0;
        double de112 = 0;
        de112 += m0diffe110;
        de112 += m0diffe110;
        double _445 = de112;
        de112 = 0;

        Qxcenteredb_ipc40[i + 1] += _445;
        double _451 = res_17_i_de;
        res_17_i_de = 0;
        mul_i102_de += (i == 0) ? _451 : 0.0;
        add_i106_de += (i == 0) ? 0 : _451;
      }
      /* invertfor.body.i109.preheader */
      /* invertcQtimesx.exit */
      // DEBUG_POINT_6
      double m0diffe91 = mul_i102_de * _cache82[ix * k + ik];
      mul_i102_de = 0;
      de35 += m0diffe91;
      de35 += m0diffe91;
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
      double _400 = de35;
      de35 = 0;

      pre_de += _400;
      de34 += 0;

      /* invertcQtimesx.exit.loopexit */
      double _362 = pre_de; // this matches
      pre_de = 0;
      Qxcenteredb_ipc40[0] += _362;
      /* mergeinvertfor.body7.i_cQtimesx.exit.loopexit */
      for (int64_t i = d - 1; i >= 0; i--) {
        /* invertfor.cond5.loopexit.i */
        if (i + 1 < d) {
          /* invertfor.cond5.loopexit.i.loopexit */
          /* mergeinvertfor.body13.i_for.cond5.loopexit.i.loopexit */
          for (int64_t j = d - 2 - i; j >= 0; j--) {
            /* invertfor.body13.i */
            // DEBUG_POINT_8
            double _305 = Qxcenteredb_ipc40[i + 1 + j];
            Qxcenteredb_ipc40[i + 1 + j] = 0;
            add21_i_de += _305;

            double _308 = add21_i_de;
            add21_i_de = 0;

            de62 += _308;
            mul20_i_de += _308;
            double m0diffe71 = mul20_i_de * _cache[ix * k * d + ik * d + i];
            printf("read cache: %.4e\n", _cache[ix * k * d + ik * d + i]);
            int64_t Lidx = (2 * d + (((int32_t)i) ^ -1)) * i / 2 + j;
            double m1diffe78 = mul20_i_de * Ls[ik * tri_size_conv + Lidx];
            mul20_i_de = 0;
            de79 += m0diffe71;
            de59 += m1diffe78;
            double _347 = de79;
            de79 = 0;
            Lsb[ik * tri_size_conv + Lidx] += _347;
            double _354 = de62;
            de62 = 0;
            Qxcenteredb_ipc40[i + 1 + j] += _354;
            // printf("_cache: %lld\n", i);
          }
          /* invertfor.body13.lr.ph.i */
          double _295 = de59;
          de59 = 0;
          xcenteredb_ipc37[i] += _295;
          /* invertfor.body7.i */
        }
      }
      /* invertfor.body7.i.preheader */
      /* mergeinvertfor.body.i114_for.body7.i.preheader */
      for (int64_t i = d - 1; i >= 0; i--) {
        /* invertfor.body.i114 */
        // DEBUG_POINT_9
        double _231 = Qxcenteredb_ipc40[0];
        Qxcenteredb_ipc40[0] = 0;
        mul_i113_de += _231;
        double m0diffe = mul_i113_de * _cache[ix * k * d + ik * d + i];
        double m1diffe = mul_i113_de * Qdiags_0[ik * d + i];
        mul_i113_de = 0;
        de50 += m0diffe;
        de51 += m1diffe;

        double _265 = de51;
        de51 = 0;
        xcenteredb_ipc37[i] += _265;
        double _271 = de50;
        de50 = 0;
        Qdiagsb_ipc21[ik * d + i] += _271;
      }
      /* invertfor.body.i114.preheader */
      // DEBUG_POINT_10
      for (int64_t i = d - 1; i >= 0; i--) {
        /* invertfor.body.i128 */
        double _208 = xcenteredb_ipc37[i];
        xcenteredb_ipc37[i] = 0;
        sub_i124_de += _208;
        double _212 = -sub_i124_de;
        sub_i124_de = 0;
        de38 += _212;
        double _215 = de38;
        de38 = 0;
        meansb[ik * d + i] += _215;
      }
      /* invertfor.body.i128.preheader */
      /* invertfor.body23 */
      double _191 = de34;
      de34 = 0;
      de35 += (ik == 0) ? 0.0 : _191;
      de32 += (ik == 0) ? _191 : 0.0;
    }
    /* invertfor.body23.lr.ph */
    /* invertfor.cond19.preheader */
    // DEBUG_POINT_11
    double _171 = de30;
    de30 = 0;
    de31 += (ix == 0) ? 0.0 : _171;
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
  free(Qdiagsb_ipc21);
  free(Qdiags_0);
}
