#include "hand.h"
#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#define NUM_RUNS 5
extern F64Descriptor3D dla_positions(/*transforms=*/double *, double *, int64_t,
                                     int64_t, int64_t, int64_t, int64_t,
                                     int64_t, int64_t,
                                     /*base_positions=*/double *, double *,
                                     int64_t, int64_t, int64_t, int64_t,
                                     int64_t,
                                     /*weights*/ double *, double *, int64_t,
                                     int64_t, int64_t, int64_t, int64_t);
extern F64Descriptor3D den_positions(/*transforms=*/double *, double *, int64_t,
                                     int64_t, int64_t, int64_t, int64_t,
                                     int64_t, int64_t,
                                     /*base_positions=*/double *, double *,
                                     int64_t, int64_t, int64_t, int64_t,
                                     int64_t,
                                     /*weights*/ double *, double *, int64_t,
                                     int64_t, int64_t, int64_t, int64_t);

double *deadbeef = (double *)0xdeadbeef;
unsigned long enzyme_hand_pos(HandInput *input, double *transforms,
                              double *ref_soln) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor3D en_res = den_positions(
      /*transforms=*/deadbeef, transforms, 0, 22, 4, 4, 16, 4, 1,
      /*base_positions=*/deadbeef, input->model.base_positions, 0,
      input->model.n_vertices, 4, 4, 1,
      /*weights=*/deadbeef, input->model.weights, 0, input->model.n_vertices,
      input->model.n_bones, input->model.n_bones, 1);

  gettimeofday(&stop, NULL);

  for (size_t i = 0; i < 22 * 4 * 4; i++) {
    ref_soln[i] = en_res.aligned[i];
  }

  free(en_res.aligned);

  return timediff(start, stop);
}

unsigned long lagrad_hand_pos(HandInput *input, double *transforms,
                              double *ref_soln) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor3D la_res = dla_positions(
      /*transforms=*/deadbeef, transforms, 0, 22, 4, 4, 16, 4, 1,
      /*base_positions=*/deadbeef, input->model.base_positions, 0,
      input->model.n_vertices, 4, 4, 1,
      /*weights=*/deadbeef, input->model.weights, 0, input->model.n_vertices,
      input->model.n_bones, input->model.n_bones, 1);
  gettimeofday(&stop, NULL);

  double err = 0.0;
  for (size_t i = 0; i < 22 * 4 * 4; i++) {
    err += fabs(ref_soln[i] - la_res.aligned[i]);
  }

  if (err > 1e-6) {
    printf("LAGrad err: %f\n", err);
  }
  free(la_res.aligned);

  return timediff(start, stop);
}

int main() {
  HandInput input = read_hand_data(false, true);
  double *transforms = (double *)malloc(22 * 4 * 4 * sizeof(double));
  random_init_d_2d(transforms, 22, 4 * 4);
  double *ref_solution = (double *)malloc(22 * 4 * 4 * sizeof(double));

  unsigned long *df = (unsigned long *)malloc(NUM_RUNS * sizeof(unsigned long));
  for (size_t run = 0; run < NUM_RUNS; run++) {
    df[run] = enzyme_hand_pos(&input, transforms, ref_solution);
  }
  printf("Enzyme:\n");
  print_ul_arr(df, NUM_RUNS);

  for (size_t run = 0; run < NUM_RUNS; run++) {
    df[run] = lagrad_hand_pos(&input, transforms, ref_solution);
  }
  printf("LAGrad:\n");
  print_ul_arr(df, NUM_RUNS);

  free(df);
  free(transforms);
  free(ref_solution);
}
