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

extern void grad_lstm_model_hb(
    /*weight=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*dweight=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*bias=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*dbias=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*hidden=*/double *, double *, int64_t, int64_t, int64_t,
    /*dhidden=*/double *, double *, int64_t, int64_t, int64_t,
    /*cell=*/double *, double *, int64_t, int64_t, int64_t,
    /*dcell=*/double *, double *, int64_t, int64_t, int64_t,
    /*input=*/double *, double *, int64_t, int64_t, int64_t,
    /*dinput=*/double *, double *, int64_t, int64_t, int64_t);

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
int lstm_model_main() {
  LSTMInput input;
  read_lstm_instance(&input);
  int b = input.b;
  size_t weight_size = 4 * b;
  double *weight = input.main_params;
  double *weightb = calloc(weight_size, sizeof(double));
  double *bias = &input.main_params[weight_size];
  double *biasb = calloc(weight_size, sizeof(double));
  double *hidden = malloc(b * sizeof(double));
  memcpy(hidden, &input.state[0], input.b * sizeof(double));
  double *hiddenb = calloc(b, sizeof(double));
  for (size_t i = 0; i < b; i++) {
    hiddenb[i] = 1.0;
  }

  double *cell = malloc(b * sizeof(double));
  memcpy(cell, &input.state[b], b * sizeof(double));
  double *cellb = calloc(b, sizeof(double));

  double *_input = &input.sequence[0];
  double *_inputb = calloc(b, sizeof(double));
  grad_lstm_model_hb(deadbeef, weight, 0, 4, b, b, 1, deadbeef, weightb, 0, 4,
                     b, b, 1, deadbeef, bias, 0, 4, b, b, 1, deadbeef, biasb, 0,
                     4, b, b, 1, deadbeef, hidden, 0, b, 1, deadbeef, hiddenb,
                     0, b, 1, deadbeef, cell, 0, b, 1, deadbeef, cellb, 0, b, 1,
                     deadbeef, _input, 0, b, 1, deadbeef, _inputb, 0, b, 1);
  // print_d_arr_2d(weightb, 4, b);
  // print_d_arr_2d(biasb, 4, b);
  // print_d_arr(cellb, b);
  // print_d_arr(_inputb, b);
  // print_d_arr(hiddenb, b);
  free_lstm_instance(&input);
  return 0;
}

extern void grad_predict_hb(
    /*w=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*dw=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*w2=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*dw2=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*s=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*ds=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t,
    /*x2=*/double *, double *, int64_t, int64_t, int64_t,
    /*dx2=*/double *, double *, int64_t, int64_t, int64_t);
extern void predict_hb(
    /*w=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*w2=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*s=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*x=*/double *, double *, int64_t, int64_t, int64_t,
    /*x2=*/double *, double *, int64_t, int64_t, int64_t);

int main() {
  LSTMInput input;
  read_lstm_instance(&input);
  int b = input.b;
  double *main_paramsb = calloc(4 * 4 * b, sizeof(double));
  double *extra_paramsb = calloc(3 * b, sizeof(double));
  double *state = malloc(2 * 2 * b * sizeof(double));
  memcpy(state, &input.state[0], 2 * 2 * b * sizeof(double));
  double *stateb = calloc(2 * 2 * b, sizeof(double));
  double *x2 = malloc(b * sizeof(double));
  double *dx2 = malloc(b * sizeof(double));
  for (size_t i = 0; i < b; i++) {
    dx2[i] = 1.0;
  }
  // predict_hb(deadbeef, input.main_params, 0, 4, 4, b, 4 * b, b, 1, deadbeef,
  //            input.extra_params, 0, 3, b, b, 1, deadbeef, state, 0, input.l, 2,
  //            b, 2 * b, b, 1, deadbeef, input.sequence, 0, b, 1, deadbeef, x2, 0,
  //            b, 1);
  // print_d_arr(x2, b);
  grad_predict_hb(deadbeef, input.main_params, 0, 4, 4, b, 4 * b, b, 1,
                  deadbeef, main_paramsb, 0, 4, 4, b, 4 * b, b, 1, deadbeef,
                  input.extra_params, 0, 3, b, b, 1, deadbeef, extra_paramsb,
                  0, 3, b, b, 1, deadbeef, state, 0, input.l, 2, b, 2 * b, b,
                  1, deadbeef, stateb, 0, input.l, 2, b, 2 * b, b, 1,
                  deadbeef, input.sequence, 0, b, 1, deadbeef, x2, 0, b, 1,
                  deadbeef, dx2, 0, b, 1);
  // print_d_arr_3d(main_paramsb, 4, 4, b);
  // print_d_arr_2d(extra_paramsb, 3, b);
  print_d_arr_2d(stateb, 4, b);
  free(main_paramsb);
  free(extra_paramsb);
  free(state);
  free(stateb);
  free(x2);
  free(dx2);
  free_lstm_instance(&input);
  return 0;
}

int benchmark_main() {
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
  return 0;
}
