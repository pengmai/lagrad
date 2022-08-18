// #include "shared_types.h"
#include "lagrad_utils.h"
// #include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// out = a - b

// void ecprint_d_arr(const double *arr, size_t n) {
//   printf("[");
//   for (int i = 0; i < n; i++) {
//     printf("%.4e", arr[i]);
//     if (i != n - 1) {
//       printf(", ");
//     }
//   }
//   printf("]\n");
// }
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

double sqnorm(int n, double const *x) {
  int i;
  double res = x[0] * x[0];
  for (i = 1; i < n; i++) {
    res = res + x[i] * x[i];
  }

  return res;
}

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

void preprocess_qs(int d, int k, double const *Qs, double *sum_qs,
                   double *Qdiags) {
  int ik, id;
  for (ik = 0; ik < k; ik++) {
    sum_qs[ik] = 0.;
    for (id = 0; id < d; id++) {
      double q = Qs[ik * d + id];
      sum_qs[ik] = sum_qs[ik] + q;
      Qdiags[ik * d + id] = exp(q);
    }
  }
}

void cQtimesx(int d, double const *Qdiag,
              double const *ltri, // strictly lower triangular part
              double const *x, double *out) {
  int i, j;
  for (i = 0; i < d; i++) {
    out[i] = Qdiag[i] * x[i];
  }

  // caching lparams as scev doesn't replicate index calculation
  // todo note changing to strengthened form
  // int Lparamsidx = 0;
  // for (i = 1; i < d; i++) {
  //   for (j = 0; j < i; j++) {
  //     int Lidx = i - 1 + (2 * d - j - 3) * j / 2;
  //     out[i] += ltri[Lidx] * x[j];
  //   }
  // }
  for (i = 0; i < d; i++) {
    // int Lparamsidx = i * (2 * d - i - 1) / 2;
    for (j = i + 1; j < d; j++) {
      // and this x
      int Lparamsidx = j - i - 1 + i * (2 * d - i - 1) / 2;
      out[j] = out[j] + ltri[Lparamsidx] * x[i];
      Lparamsidx++;
    }
  }
}

void cQtimesx_adjoint(int d, double const *Ls, double const *dQxcentered,
                      double const *xcentered, double *dxcentered,
                      double *Lsb) {
  for (int i = 0; i < d; i++) {
    // int Lparamsidx = i * (2 * d - i - 1) / 2;
    for (int j = i + 1; j < d; j++) {
      // and this x
      int Lparamsidx = j - i - 1 + i * (2 * d - i - 1) / 2;
      dxcentered[i] += Ls[Lparamsidx] * dQxcentered[j];
      Lsb[Lparamsidx] += dQxcentered[j] * xcentered[i];
      Lparamsidx++;
    }
  }
}

/* trying this wrapper so MLIR's handbuf implementation can call into this. */
void _mlir_ciface_bQtimesx_adjoint(F64Descriptor1D *L,
                                   F64Descriptor1D *dQxcentered,
                                   F64Descriptor1D *xcentered,
                                   F64Descriptor1D *dxcentered,
                                   F64Descriptor1D *Lsb) {
  cQtimesx_adjoint(dQxcentered->size, &L->aligned[L->offset],
                   dQxcentered->aligned, &xcentered->aligned[xcentered->offset],
                   dxcentered->aligned, &Lsb->aligned[Lsb->offset]);
}

void ec_main_term(int d, int k, int n, double const *__restrict alphas,
                  double const *__restrict means, double const *__restrict Qs,
                  double const *__restrict Ls, double const *__restrict x,
                  double *err) {
#define int int64_t
  int ix, ik;
  int tri_size = d * (d - 1) / 2;
  double *Qdiags = (double *)malloc(d * k * sizeof(double));
  double *sum_qs = (double *)malloc(k * sizeof(double));
  double *xcentered = (double *)malloc(d * sizeof(double));
  double *Qxcentered = (double *)malloc(d * sizeof(double));
  double *main_term = (double *)malloc(k * sizeof(double));

  preprocess_qs(d, k, Qs, &sum_qs[0], &Qdiags[0]);

  double slse = 0.;
  for (ix = 0; ix < n; ix++) {
    for (ik = 0; ik < k; ik++) {
      subtract(d, &x[ix * d], &means[ik * d], &xcentered[0]);
      cQtimesx(d, &Qdiags[ik * d], &Ls[ik * tri_size], &xcentered[0],
               &Qxcentered[0]);
      main_term[ik] = alphas[ik] + sum_qs[ik] - 0.5 * sqnorm(d, &Qxcentered[0]);
    }

    slse = slse + log_sum_exp(k, &main_term[0]);
  }

  *err = slse;

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

void enzyme_c_main_term(int d, int k, int n, double *alphas, double *alphasb,
                        double *means, double *meansb, double *Qs, double *Qsb,
                        double *Ls, double *Lsb, double *x) {
  double out = 0.0, dout = 1.0;
  __enzyme_autodiff(ec_main_term, d, k, n, alphas, alphasb, means, meansb, Qs,
                    Qsb, Ls, Lsb, enzyme_const, x, &out, &dout);
}

void cvecmat(int d, double *x, const double *ltri, double *tmp, double *out) {
  // cblas_dgemv(CblasRowMajor, CblasTrans, d, d, 1.0, tmp, d, x, 1, 1.0, out,
  // 1);
  for (int i = 0; i < d; i++) {
    int Lparamsidx = i * (2 * d - i - 1) / 2;
    for (int j = i + 1; j < d; j++) {
      // and this x
      out[i] = out[i] + ltri[Lparamsidx] * x[j];
      Lparamsidx++;
    }
  }
}

// void blasQtimesx(int d, double const *Qdiag,
//                  double const *ltri, // strictly lower triangular part
//                  double const *x, double *out) {
//   int i, j;
//   for (i = 0; i < d; i++) {
//     out[i] = Qdiag[i] * x[i];
//   }
//   cblas_dgemv(CblasRowMajor, CblasNoTrans, d, d, 1.0, ltri, d, x, 1, 1.0,
//   out,
//               1);
// }

void couter(int d, double *x, double *y, double *out) {
  // int Lparamsidx = 0;
  for (int i = 0; i < d; i++) {
    int Lparamsidx = i * (2 * d - i - 1) / 2;
    for (int j = i + 1; j < d; j++) {
      // and this x
      out[Lparamsidx] += x[j] * y[i];
      Lparamsidx++;
    }
  }
}

void cgrad_log_sum_exp(int d, double const *x, double g, double *restrict dx) {
  double lse = log_sum_exp(d, x);

  for (size_t i = 0; i < d; i++) {
    double e = exp(x[i] - lse);
    // if (fabs(e) < 1e-32) {
    //   e = 0;
    // }
    dx[i] = g * e;
  }
}

void materialize_L(int k, int d, double *Lcompressed, double *Lfull) {
  int icf_idx;
  int tri_size = d * (d - 1) / 2;
  for (int i = 0; i < k; i++) {
    icf_idx = 0;
    for (int j = 0; j < d; j++) {
      for (int l = j + 1; l < d; l++) {
        Lfull[i * (d * d) + l * d + j] = Lcompressed[i * tri_size + icf_idx];
        icf_idx++;
      }
    }
  }
}

void manual_c_main_term(int d, int k, int n, double const *restrict alphas,
                        double *restrict alphasb, double const *restrict means,
                        double *restrict meansb, double const *restrict Qs,
                        double *restrict Qsb, double const *restrict Ls,
                        double *restrict Lsb, double const *restrict x) {
#define int int64_t
  int ix, ik;
  int tri_size = d * (d - 1) / 2;
  double *Qdiags = (double *)malloc(d * k * sizeof(double));
  double *dQdiags = (double *)calloc(d * k, sizeof(double));
  double *sum_qs = (double *)malloc(k * sizeof(double));
  double *dsum_qs = (double *)calloc(k, sizeof(double));
  double *xcentered = (double *)malloc(d * sizeof(double));
  double *dxcentered = (double *)malloc(d * sizeof(double));
  double *Qxcentered = (double *)malloc(d * sizeof(double));
  double *dQxcentered = (double *)malloc(d * sizeof(double));
  double *main_term = (double *)malloc(k * sizeof(double));
  double *dmain_term = (double *)malloc(k * sizeof(double));

  preprocess_qs(d, k, Qs, &sum_qs[0], &Qdiags[0]);

  double slse = 0.;
  double *xcentered_cache = malloc(k * d * sizeof(double));
  double *Qxcentered_cache = malloc(k * d * sizeof(double));
  // double *vecmat_temp = calloc(k * d * d, sizeof(double));
  // Transform L
  // materialize_L(k, d, Ls, vecmat_temp);

  for (ix = 0; ix < n; ix++) {
    for (ik = 0; ik < k; ik++) {
      subtract(d, &x[ix * d], &means[ik * d], &xcentered[0]);
      // blasQtimesx(d, &Qdiags[ik * d], &vecmat_temp[ik * d * d],
      // &xcentered[0],
      //             &Qxcentered[0]);
      cQtimesx(d, &Qdiags[ik * d], &Ls[ik * tri_size], &xcentered[0],
               &Qxcentered[0]);

      memcpy(&xcentered_cache[ik * d], xcentered, d * sizeof(double));
      memcpy(&Qxcentered_cache[ik * d], Qxcentered, d * sizeof(double));
      main_term[ik] = alphas[ik] + sum_qs[ik] - 0.5 * sqnorm(d, &Qxcentered[0]);
      // What the flying fuck??
      // dmain_term[ik] = 1e-20;
    }

    // slse = slse + log_sum_exp(k, &main_term[0]);
    cgrad_log_sum_exp(k, main_term, 1.0, dmain_term);

    for (ik = 0; ik < k; ik++) {
      alphasb[ik] += dmain_term[ik];
      dsum_qs[ik] += dmain_term[ik];
    }

    for (ik = 0; ik < k; ik++) {
      // double *xcentered = &xcentered_cache[ik * d];
      // double *Qxcentered = &Qxcentered_cache[ik * d];
      memcpy(xcentered, &xcentered_cache[ik * d], d * sizeof(double));
      memcpy(Qxcentered, &Qxcentered_cache[ik * d], d * sizeof(double));
      for (int i = 0; i < d; i++) {
        dQxcentered[i] = -dmain_term[ik] * Qxcentered[i];
        dxcentered[i] = Qdiags[ik * d + i] * dQxcentered[i];
      }

      // // cvecmat(d, dQxcentered, &Ls[ik * tri_size], &vecmat_temp[ik * d *
      // d],
      // //         dxcentered);
      // // couter(d, dQxcentered, xcentered, &Lsb[ik * tri_size]);
      // // int Lparamsidx = 0;
      // Fusion is important, makes it 10% faster.
      cQtimesx_adjoint(d, &Ls[ik * tri_size], dQxcentered, xcentered,
                       dxcentered, &Lsb[ik * tri_size]);
      for (int i = 0; i < d; i++) {
        meansb[ik * d + i] -= dxcentered[i];
        dQdiags[ik * d + i] += dQxcentered[i] * xcentered[i];
      }
    }
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < d; j++) {
      Qsb[i * d + j] = dsum_qs[i] + dQdiags[i * d + j] * Qdiags[i * d + j];
    }
  }

  // free(vecmat_temp);
  free(xcentered_cache);
  free(Qxcentered_cache);

  free(Qdiags);
  free(dQdiags);
  free(sum_qs);
  free(dsum_qs);
  free(xcentered);
  free(dxcentered);
  free(Qxcentered);
  free(dQxcentered);
  free(main_term);
  free(dmain_term);

#undef int
}
