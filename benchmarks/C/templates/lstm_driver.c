#include "lstm.h"
#include "mlir_c_abi.h"
#include <math.h>

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

extern LSTMGrad
lagrad_lstm(/*main_params=*/double *, double *, int64_t, int64_t, int64_t,
            /*extra_params=*/double *, double *, int64_t, int64_t, int64_t,
            /*state=*/double *, double *, int64_t, int64_t, int64_t,
            /*sequence=*/double *, double *, int64_t, int64_t, int64_t);

double *deadbeef = (double *)0xdeadbeef;
int main() {
  LSTMInput input;
  read_lstm_instance(&input);

  double *state = malloc(input.state_sz * sizeof(double));
  for (size_t i = 0; i < input.state_sz; i++) {
    state[i] = input.state[i];
  }

  // double loss = 0.0;
  // lstm_objective(input.l, input.c, input.b, input.main_params,
  //                input.extra_params, state, input.sequence, &loss);
  // printf("C Primal: %.8e\n", loss);

  // loss = 0.0;
  // double dloss = 1.0;
  // double *dmain_params = (double *)malloc(input.main_sz * sizeof(double));
  // double *dextra_params = (double *)malloc(input.extra_sz * sizeof(double));
  // for (size_t i = 0; i < input.main_sz; i++) {
  //   dmain_params[i] = 0;
  // }
  // for (size_t i = 0; i < input.extra_sz; i++) {
  //   dextra_params[i] = 0;
  // }
  // for (size_t i = 0; i < input.state_sz; i++) {
  //   state[i] = input.state[i];
  // }

  // enzyme_c_lstm_objective(input.l, input.c, input.b, input.main_params,
  //                         dmain_params, input.extra_params, dextra_params,
  //                         state, input.sequence, &loss, &dloss);

  // printf("Didn't segfault\n");
  // print_d_arr(dextra_params, 10);
  for (size_t i = 0; i < input.state_sz; i++) {
    state[i] = input.state[i];
  }
  // double mlir_p = mlstm_objective(
  //     deadbeef, input.main_params, 0, input.main_sz, 1, deadbeef,
  //     input.extra_params, 0, input.extra_sz, 1, deadbeef, state, 0,
  //     input.state_sz, 1, deadbeef, input.sequence, 0, input.seq_sz, 1);
  // printf("MLIR Primal: %.8e\n", mlir_p);

  LSTMGrad res = lagrad_lstm(deadbeef, input.main_params, 0, input.main_sz, 1,
                             deadbeef, input.extra_params, 0, input.extra_sz, 1,
                             deadbeef, state, 0, input.state_sz, 1, deadbeef,
                             input.sequence, 0, input.seq_sz, 1);
  print_d_arr(res.dmain_params.aligned, 10);
  free(res.dmain_params.aligned);
  free(res.dextra_params.aligned);
  // free_lstm_instance(&input);
}
