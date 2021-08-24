#include "mlir_c_abi.h"
#include "shared_types.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#define NUM_WARMUPS {{ num_warmups }}
#define NUM_RUNS {{ num_runs }}
#define N {{ n }}
#define M {{ m }}

// #define NUM_WARMUPS 10
// #define NUM_RUNS 10
// #define M 256
// #define N 256

// {% if args == [0] %}
extern F32Descriptor2D grad_matvec_first(/*A=*/float *, float *, int64_t,
                                         int64_t, int64_t, int64_t, int64_t,
                                         /*x=*/float *, float *, int64_t,
                                         int64_t, int64_t);

extern F32Descriptor2D blas_grad_matvec_first(/*A=*/float *, float *, int64_t,
                                              int64_t, int64_t, int64_t,
                                              int64_t,
                                              /*x=*/float *, float *, int64_t,
                                              int64_t, int64_t);
extern float *enzyme_matvec_first(float *, float *, int64_t, int64_t);
extern float *c_matvec_first(float *, float *, float *, int64_t, int64_t);
extern float *openblas_matvec_first(float *, float *, float *, int64_t,
                                    int64_t);
// {% endif %}

void check_first_arg(float *da, float *x, size_t m, size_t n,
                     const char *application) {
  float total_err = 0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      total_err += da[i * n + j] - x[j];
    }
  }
  if (abs(total_err) > 1e-9) {
    printf("(%s) matvec err: %f\n", application, total_err);
  }
}

int main() {
  unsigned long c_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long blas_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long enzyme_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long grad_loop_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long grad_blas_results[NUM_WARMUPS + NUM_RUNS];
  float *A = (float *)malloc(M * N * sizeof(float));
  float *x = (float *)malloc(N * sizeof(float));
  float *g = (float *)malloc(M * sizeof(float));

  random_init_2d(A, M, N);
  random_init(x, N);
  uniform_init(1.0f, g, M);

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dA = enzyme_matvec_first(A, x, M, N);
    // {% endif %}
    gettimeofday(&stop, NULL);
    enzyme_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dA, x, M, N, "enzyme");
    free(dA);
    // {% endif %}
  }

  float *deadbeef = (float *)0xdeadbeef;
  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    F32Descriptor2D dA =
        grad_matvec_first(deadbeef, A, 0, M, N, 1, 1, deadbeef, x, 0, N, 1);
    // {% endif %}
    gettimeofday(&stop, NULL);

    grad_loop_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dA.aligned, x, M, N, "grad_loops");
    free(dA.aligned);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    F32Descriptor2D dA = blas_grad_matvec_first(deadbeef, A, 0, M, N, 1, 1,
                                                deadbeef, x, 0, N, 1);
    // {% endif %}
    gettimeofday(&stop, NULL);

    grad_blas_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dA.aligned, x, M, N, "grad_blas");
    free(dA.aligned);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dA = openblas_matvec_first(A, x, g, M, N);
    // {% endif %}
    gettimeofday(&stop, NULL);
    blas_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dA, x, M, N, "handwritten_blas");
    free(dA);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dA = c_matvec_first(A, x, g, M, N);
    // {% endif %}
    gettimeofday(&stop, NULL);
    c_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dA, x, M, N, "handwritten_c");
    free(dA);
    // {% endif %}
  }

  printf("grad_loops: ");
  print_ul_arr(grad_loop_results + NUM_WARMUPS, NUM_RUNS);
  printf("grad_blas: ");
  print_ul_arr(grad_blas_results + NUM_WARMUPS, NUM_RUNS);
  printf("enzyme: ");
  print_ul_arr(enzyme_results + NUM_WARMUPS, NUM_RUNS);
  printf("handwritten_c: ");
  print_ul_arr(c_results + NUM_WARMUPS, NUM_RUNS);
  printf("handwritten_blas: ");
  print_ul_arr(blas_results + NUM_WARMUPS, NUM_RUNS);
}
