#include "hand.h"
#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#define NUM_RUNS 5

extern F64Descriptor2D la_hand_err(/*positions=*/double *, double *, int64_t,
                                   int64_t, int64_t, int64_t, int64_t,
                                   /*points=*/double *, double *, int64_t,
                                   int64_t, int64_t, int64_t, int64_t,
                                   /*corresp=*/int32_t *, int32_t *, int64_t,
                                   int64_t, int64_t);
extern F64Descriptor2D den_hand_err(/*arg0=*/double *, double *, int64_t,
                                    int64_t, int64_t, int64_t, int64_t,
                                    /*arg1=*/double *, double *, int64_t,
                                    int64_t, int64_t, int64_t, int64_t,
                                    /*arg2=*/int32_t *, int32_t *, int64_t,
                                    int64_t, int64_t);
extern F64Descriptor2D dla_hand_err(/*positions=*/double *, double *, int64_t,
                                    int64_t, int64_t, int64_t, int64_t,
                                    /*points=*/double *, double *, int64_t,
                                    int64_t, int64_t, int64_t, int64_t,
                                    /*corresp=*/int32_t *, int32_t *, int64_t,
                                    int64_t, int64_t);

double *deadbeef = (double *)0xdeadbeef;
unsigned long enzyme_hand_err(HandInput *input, double *positions,
                              double *ref_soln) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor2D en_res = den_hand_err(
      /*positions=*/deadbeef, positions, 0, 544, 3, 3, 1,
      /*points=*/deadbeef, input->points, 0, 100, 3, 3, 1,
      /*corresp=*/(int32_t *)deadbeef, input->correspondences, 0, 100, 1);

  gettimeofday(&stop, NULL);

  for (size_t i = 0; i < en_res.size_0; i++) {
    for (size_t j = 0; j < en_res.size_1; j++) {
      ref_soln[i * en_res.size_1 + j] = en_res.aligned[i * en_res.size_1 + j];
    }
  }

  free(en_res.aligned);

  return timediff(start, stop);
}

unsigned long lagrad_hand_err(HandInput *input, double *positions,
                              double *ref_soln) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  F64Descriptor2D la_res = dla_hand_err(
      /*positions=*/deadbeef, positions, 0, 544, 3, 3, 1,
      /*points=*/deadbeef, input->points, 0, 100, 3, 3, 1,
      /*corresp=*/(int32_t *)deadbeef, input->correspondences, 0, 100, 1);
  gettimeofday(&stop, NULL);

  double err = 0.0;
  for (size_t i = 0; i < la_res.size_0; i++) {
    for (size_t j = 0; j < la_res.size_1; j++) {
      err += fabs(ref_soln[i * la_res.size_1 + j] -
                  la_res.aligned[i * la_res.size_1 + j]);
    }
  }
  if (err > 1e-6) {
    printf("LAGrad err: %f\n", err);
  }
  free(la_res.aligned);

  return timediff(start, stop);
}

int main() {
  HandInput input = read_hand_data(false, true);
  double *positions = (double *)malloc(544 * 3 * sizeof(double));
  random_init_d_2d(positions, 544, 3);
  double *ref_solution = (double *)malloc(544 * 3 * sizeof(double));

  unsigned long took = enzyme_hand_err(&input, positions, ref_solution);
  printf("Enzyme took: %lu\n", took);
  took = lagrad_hand_err(&input, positions, ref_solution);
  printf("LAGrad took: %lu\n", took);

  free(positions);
  free(ref_solution);
}
