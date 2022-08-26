#include "hand.h"
#include "lagrad_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUM_RUNS 6
#define CHECK_MEM 0

typedef F64Descriptor1D (*hand_jacobian_row)(HandInput *input, double *derr);
typedef HandComplicatedGrad (*HandComplicatedJacobianRow)(HandInput *input,
                                                          double *derr);

void hand_jacobian_simple(hand_jacobian_row compute_row, HandInput *input,
                          double *J) {
  int err_size = 3 * input->n_pts;
  double *derr = (double *)malloc(err_size * sizeof(double));
  for (size_t i = 0; i < err_size; i++) {
    for (size_t j = 0; j < err_size; j++) {
      derr[j] = (i == j) ? 1.0 : 0.0;
    }
    F64Descriptor1D dtheta = compute_row(input, derr);
    for (size_t j = 0; j < input->n_theta; j++) {
      J[i * input->n_theta + j] = dtheta.aligned[j];
    }
    free(dtheta.aligned);
  }
  free(derr);
}

void hand_jacobian_complicated(HandComplicatedJacobianRow compute_row,
                               HandInput *input, double *J) {
  int err_size = 3 * input->n_pts;
  int J_stride = input->n_theta + 2;
  double *derr = malloc(err_size * sizeof(double));
  for (size_t i = 0; i < err_size; i++) {
    for (size_t j = 0; j < err_size; j++) {
      derr[j] = (i == j) ? 1.0 : 0.0;
    }
    HandComplicatedGrad res = compute_row(input, derr);

    // Write theta part
    for (size_t j = 0; j < res.dtheta.size; j++) {
      J[i * J_stride + 2 + j] = res.dtheta.aligned[j];
    }
    // Write us part
    for (size_t j = 0; j < 2; j++) {
      J[i * J_stride + j] = res.dus.aligned[(i / 3) * 2 + j];
    }
    free(res.dtheta.aligned);
    free(res.dus.aligned);
  }
  free(derr);
}

typedef struct HandApp {
  const char *name;
  hand_jacobian_row row_func;
  HandComplicatedJacobianRow complicated_row_func;
} HandApp;

unsigned long collect_hand(HandApp app, HandInput *input, double *J,
                           double *ref_J, bool complicated) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  if (complicated) {
    hand_jacobian_complicated(app.complicated_row_func, input, J);
  } else {
    hand_jacobian_simple(app.row_func, input, J);
  }
  gettimeofday(&stop, NULL);
  if (CHECK_MEM) {
  } else {
    verify_hand_results(ref_J, J, 3 * input->n_pts, input->n_theta, app.name);
  }
  return timediff(start, stop);
}

void populate_ref(HandInput *input, double *ref_J, bool complicated) {
  if (complicated) {
    hand_jacobian_complicated(lagrad_hand_complicated, input, ref_J);
  } else {
    hand_jacobian_simple(lagrad_hand_simple, input, ref_J);
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <model-path> <data-file> <complicated>",
            argv[0]);
    return 1;
  }
  bool complicated = strtol(argv[3], NULL, 10);
  HandInput input = read_hand_data(argv[1], argv[2], complicated, true);
  int J_rows = 3 * input.n_pts;
  int J_cols = complicated ? input.n_theta + 2 : input.n_theta;
  double *ref_J = (double *)malloc(J_rows * J_cols * sizeof(double));
  if (!CHECK_MEM) {
    populate_ref(&input, ref_J, complicated);
  }
  double *J = (double *)malloc(J_rows * J_cols * sizeof(double));
  HandApp apps[] = {
      {.name = "LAGrad",
       .row_func = lagrad_hand_simple,
       .complicated_row_func = lagrad_hand_complicated},
      // {.name = "Enzyme/MLIR",
      //  .row_func = enzyme_mlir_hand_simple,
      //  .complicated_row_func = enzyme_mlir_hand_complicated},
      // {.name = "Enzyme/C", .row_func = enzyme_c_hand_simple},
  };
  size_t num_apps = sizeof(apps) / sizeof(apps[0]);
  unsigned long results_df[NUM_RUNS];
  for (size_t app = 0; app < num_apps; app++) {
    printf("%s: ", apps[app].name);
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = collect_hand(apps[app], &input, J, ref_J, complicated);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  free(ref_J);
  free(J);
}
