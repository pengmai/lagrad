#include "lstm.h"
#include "mlir_c_abi.h"
#include <math.h>
#include <string.h>

#define NUM_RUNS 6

typedef struct {
  F64Descriptor1D dmain_params, dextra_params;
} LSTMGrad;

extern void
lstm_objective(int l, int c, int b, double const *__restrict main_params,
               double const *__restrict extra_params, double *__restrict state,
               double const *__restrict sequence, double *__restrict loss);
extern void
enzyme_c_lstm_objective(int l, int c, int b, double const *main_params,
                        double *dmain_params, double const *extra_params,
                        double *dextra_params, double *state,
                        double const *sequence, double *loss, double *dloss);
extern double
mlstm_objective(/*main_params=*/double *, double *, int64_t, int64_t, int64_t,
                /*extra_params=*/double *, double *, int64_t, int64_t, int64_t,
                /*state=*/double *, double *, int64_t, int64_t, int64_t,
                /*sequence=*/double *, double *, int64_t, int64_t, int64_t);

extern double
elstm_objective(/*main_params=*/double *, double *, int64_t, int64_t, int64_t,
                /*extra_params=*/double *, double *, int64_t, int64_t, int64_t,
                /*state=*/double *, double *, int64_t, int64_t, int64_t,
                /*sequence=*/double *, double *, int64_t, int64_t, int64_t,
                /*out=*/double *, double *, int64_t);

extern LSTMGrad
lagrad_lstm(/*main_params=*/double *, double *, int64_t, int64_t, int64_t,
            /*extra_params=*/double *, double *, int64_t, int64_t, int64_t,
            /*state=*/double *, double *, int64_t, int64_t, int64_t,
            /*sequence=*/double *, double *, int64_t, int64_t, int64_t);

void verify_lstm_jacobian(int main_sz, int extra_sz, double *dmain_params,
                          double *dextra_params, double *ref_jacobian,
                          const char *application) {
  double err = 0.0;
  for (size_t i = 0; i < main_sz; i++) {
    err += fabs(dmain_params[i] - ref_jacobian[i]);
  }
  if (err > 1e-8)
    printf("(%s) main params error: %f\n", application, err);

  err = 0.0;
  for (size_t i = 0; i < extra_sz; i++) {
    err += fabs(dextra_params[i] - ref_jacobian[main_sz + i]);
  }
  if (err > 1e-8)
    printf("(%s) extra params error: %f\n", application, err);
}

unsigned long collect_enzyme_c_lstm(LSTMInput input, double *state,
                                    double *ref_jacobian) {
  struct timeval start, stop;
  memcpy(state, input.state, input.state_sz * sizeof(double));

  gettimeofday(&start, NULL);
  double loss = 0.0, dloss = 1.0;
  double *dmain_params = calloc(input.main_sz, sizeof(double));
  double *dextra_params = calloc(input.extra_sz, sizeof(double));
  enzyme_c_lstm_objective(input.l, input.c, input.b, input.main_params,
                          dmain_params, input.extra_params, dextra_params,
                          state, input.sequence, &loss, &dloss);
  gettimeofday(&stop, NULL);

  verify_lstm_jacobian(input.main_sz, input.extra_sz, dmain_params,
                       dextra_params, ref_jacobian, "Enzyme/C");
  free(dmain_params);
  free(dextra_params);
  return timediff(start, stop);
}

double *deadbeef = (double *)0xdeadbeef;
int main() {
  LSTMInput input;
  read_lstm_instance(&input);
  double *ref_jacobian =
      malloc((input.main_sz + input.extra_sz) * sizeof(double));
  read_ref_grad("ref_lstm_grad.txt", input.main_sz, input.extra_sz,
                ref_jacobian);

  double *state = malloc(input.state_sz * sizeof(double));
  unsigned long results_df[NUM_RUNS];
  for (size_t run = 0; run < NUM_RUNS; run++) {

    results_df[run] = collect_enzyme_c_lstm(input, state, ref_jacobian);
  }
  print_ul_arr(results_df, NUM_RUNS);

  for (size_t i = 0; i < input.state_sz; i++) {
    state[i] = input.state[i];
  }

  // double loss = 0.0;
  // lstm_objective(input.l, input.c, input.b, input.main_params,
  //                input.extra_params, state, input.sequence, &loss);
  // printf("C Primal: %.8e\n", loss);

  // double mlir_p = mlstm_objective(
  //     deadbeef, input.main_params, 0, input.main_sz, 1, deadbeef,
  //     input.extra_params, 0, input.extra_sz, 1, deadbeef, state, 0,
  //     input.state_sz, 1, deadbeef, input.sequence, 0, input.seq_sz, 1);
  // printf("MLIR Primal: %.8e\n", mlir_p);

  // double err = 0.0;
  // for (size_t i = 0; i < input.state_sz; i++) {
  //   state[i] = input.state[i];
  // }
  // double mlir_enzyme_p =
  //     elstm_objective(deadbeef, input.main_params, 0, input.main_sz, 1,
  //                     deadbeef, input.extra_params, 0, input.extra_sz, 1,
  //                     deadbeef, state, 0, input.state_sz, 1, deadbeef,
  //                     input.sequence, 0, input.seq_sz, 1, deadbeef, &err, 0);
  // printf("Enzyme/MLIR primal: %.8e\n", mlir_enzyme_p);

  // LSTMGrad res = lagrad_lstm(deadbeef, input.main_params, 0, input.main_sz,
  // 1,
  //                            deadbeef, input.extra_params, 0, input.extra_sz,
  //                            1, deadbeef, state, 0, input.state_sz, 1,
  //                            deadbeef, input.sequence, 0, input.seq_sz, 1);
  // print_d_arr(res.dmain_params.aligned, 10);
  // free(res.dmain_params.aligned);
  // free(res.dextra_params.aligned);
  free(ref_jacobian);
  free(state);
  free_lstm_instance(&input);
}
