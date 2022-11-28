#include "lagrad_utils.h"
#include "lstm.h"
#include "memusage.h"
#include <math.h>
#include <string.h>
#include <unistd.h>

#define NUM_RUNS 6
/* set CHECK_MEM to 1 to measure memory consumption, 0 to measure performance */
#define CHECK_MEM 0

double *deadbeef = (double *)0xdeadbeef;
RunProcDyn rpd;
void check_mem_usage() {
  run_get_dynamic_proc_info(getpid(), &rpd);
  printf("%zu\t%zu\n", rpd.rss, rpd.vsize);
}

typedef struct {
  F64Descriptor4D dmain_params;
  F64Descriptor2D dextra_params;
} LSTMGrad;

extern void grad_lstm_objective_hb(
    /*main_params=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t,
    /*dmain_params=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t,
    /*extra_params=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*dextra_params=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*state=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t,
    /*sequence=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*g=*/double);

extern void
lstm_objective(int l, int c, int b, double const *__restrict main_params,
               double const *__restrict extra_params, double *__restrict state,
               double const *__restrict sequence, double *__restrict loss);
extern void
enzyme_c_lstm_objective(int l, int c, int b, double const *main_params,
                        double *dmain_params, double const *extra_params,
                        double *dextra_params, double *state,
                        double const *sequence, double *loss, double *dloss);
extern double mlstm_objective(/*main_params=*/double *, double *, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int64_t,
                              /*extra_params=*/double *, double *, int64_t,
                              int64_t, int64_t, int64_t, int64_t,
                              /*state=*/double *, double *, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t,
                              /*sequence=*/double *, double *, int64_t, int64_t,
                              int64_t, int64_t, int64_t);

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

extern LSTMGrad lagrad_lstm(/*main_params=*/double *, double *, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t,
                            /*extra_params=*/double *, double *, int64_t,
                            int64_t, int64_t, int64_t, int64_t,
                            /*state=*/double *, double *, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            /*sequence=*/double *, double *, int64_t, int64_t,
                            int64_t, int64_t, int64_t);
extern LSTMGrad enzyme_mlir_lstm(/*main_params=*/double *, double *, int64_t,
                                 int64_t, int64_t, int64_t, int64_t, int64_t,
                                 int64_t, int64_t, int64_t,
                                 /*extra_params=*/double *, double *, int64_t,
                                 int64_t, int64_t, int64_t, int64_t,
                                 /*state=*/double *, double *, int64_t, int64_t,
                                 int64_t, int64_t, int64_t, int64_t, int64_t,
                                 /*sequence=*/double *, double *, int64_t,
                                 int64_t, int64_t, int64_t, int64_t);

LSTMGrad lagrad_lstm_wrapper(LSTMInput *input, double *state) {
  int l = input->l, c = input->c, b = input->b;
  return lagrad_lstm(
      /*main_params=*/deadbeef, input->main_params, 0, l, 2, 4, b, 2 * 4 * b,
      4 * b, b, 1,
      /*extra_params*/ deadbeef, input->extra_params, 0, 3, b, b, 1,
      /*state=*/deadbeef, state, 0, l, 2, b, 2 * b, b, 1,
      /*sequence=*/deadbeef, input->sequence, 0, c, b, b, 1);
}

// LSTMGrad enzyme_mlir_lstm_wrapper(LSTMInput *input, double *state) {
//   int l = input->l, c = input->c, b = input->b;
//   return enzyme_mlir_lstm(
//       /*main_params=*/deadbeef, input->main_params, 0, l, 2, 4, b, 2 * 4 * b,
//       4 * b, b, 1,
//       /*extra_params*/ deadbeef, input->extra_params, 0, 3, b, b, 1,
//       /*state=*/deadbeef, state, 0, l, 2, b, 2 * b, b, 1,
//       /*sequence=*/deadbeef, input->sequence, 0, c, b, b, 1);
// }

LSTMGrad enzyme_c_lstm_wrapper(LSTMInput *input, double *state) {
  int l = input->l, c = input->c, b = input->b;
  double loss = 0.0, dloss = 1.0;
  F64Descriptor4D dmain_params = {.allocated = NULL,
                                  .aligned =
                                      calloc(input->main_sz, sizeof(double)),
                                  .offset = 0,
                                  .size_0 = l,
                                  .size_1 = 2,
                                  .size_2 = 4,
                                  .size_3 = b,
                                  .stride_0 = 2 * 4 * b,
                                  .stride_1 = 4 * b,
                                  .stride_2 = b,
                                  .stride_3 = 1};
  F64Descriptor2D dextra_params = {.allocated = NULL,
                                   .aligned =
                                       calloc(input->extra_sz, sizeof(double)),
                                   .offset = 0,
                                   .size_0 = c,
                                   .size_1 = b,
                                   .stride_0 = b,
                                   .stride_1 = 1};
  enzyme_c_lstm_objective(l, c, b, input->main_params, dmain_params.aligned,
                          input->extra_params, dextra_params.aligned, state,
                          input->sequence, &loss, &dloss);
  LSTMGrad res = {.dmain_params = dmain_params, .dextra_params = dextra_params};
  return res;
}

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

typedef struct LSTMApp {
  const char *name;
  LSTMGrad (*func)(LSTMInput *input, double *state);
} LSTMApp;

unsigned long collect_lstm(LSTMApp app, LSTMInput *input, double *state,
                           double *ref_jacobian) {
  struct timeval start, stop;
  memcpy(state, input->state, input->state_sz * sizeof(double));

  gettimeofday(&start, NULL);
  LSTMGrad res = app.func(input, state);
  gettimeofday(&stop, NULL);
  if (CHECK_MEM) {
    check_mem_usage();
  } else {
    verify_lstm_jacobian(input->main_sz, input->extra_sz,
                         res.dmain_params.aligned, res.dextra_params.aligned,
                         ref_jacobian, app.name);
  }
  free(res.dmain_params.aligned);
  free(res.dextra_params.aligned);
  return timediff(start, stop);
}

void populate_ref_grad(LSTMInput input, double *state, double *ref_jacobian) {
  memcpy(state, input.state, input.state_sz * sizeof(double));

  double loss = 0.0, dloss = 1.0;
  double *dmain_params = calloc(input.main_sz, sizeof(double));
  double *dextra_params = calloc(input.extra_sz, sizeof(double));
  enzyme_c_lstm_objective(input.l, input.c, input.b, input.main_params,
                          dmain_params, input.extra_params, dextra_params,
                          state, input.sequence, &loss, &dloss);

  for (size_t i = 0; i < input.main_sz; i++) {
    ref_jacobian[i] = dmain_params[i];
  }
  for (size_t i = 0; i < input.extra_sz; i++) {
    ref_jacobian[input.main_sz + i] = dextra_params[i];
  }

  free(dmain_params);
  free(dextra_params);
}

int main() {
  LSTMInput input;
  read_lstm_instance("{{data_file}}", &input);
  double *ref_jacobian =
      malloc((input.main_sz + input.extra_sz) * sizeof(double));
  double *state = malloc(input.state_sz * sizeof(double));
  if (!CHECK_MEM) {
    populate_ref_grad(input, state, ref_jacobian);
  }
  LSTMApp apps[] = {
      //
      {.name = "LAGrad", .func = lagrad_lstm_wrapper},
      // {.name = "Enzyme/MLIR", .func = enzyme_mlir_lstm_wrapper},
      // {.name = "Enzyme/C", .func = enzyme_c_lstm_wrapper}
  };
  size_t num_apps = sizeof(apps) / sizeof(apps[0]);
  unsigned long results_df[NUM_RUNS];
  for (size_t app = 0; app < num_apps; app++) {
    printf("%s: ", apps[app].name);
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = collect_lstm(apps[app], &input, state, ref_jacobian);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free(state);
  free_lstm_instance(&input);
  return 0;
}
