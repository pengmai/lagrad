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
extern F32Descriptor1D grad_vecmat_first(/*x=*/float *, float *, int64_t,
                                         int64_t, int64_t,
                                         /*A=*/float *, float *, int64_t,
                                         int64_t, int64_t, int64_t, int64_t);
extern F32Descriptor1D blas_grad_vecmat_first(/*x=*/float *, float *, int64_t,
                                              int64_t, int64_t,
                                              /*A=*/float *, float *, int64_t,
                                              int64_t, int64_t, int64_t,
                                              int64_t);
extern float *enzyme_vecmat_first(float *, float *, int64_t, int64_t);
extern float *c_vecmat_first(float *, float *, float *, int64_t, int64_t);
extern float *openblas_vecmat_first(float *, float *, float *, int64_t,
                                    int64_t);
// {% endif %}

void check_first_arg(float *dx, float *A, size_t m, size_t n,
                     const char *application) {
  float total_err = 0;
  for (size_t i = 0; i < m; i++) {
    total_err += dx[i];
    for (size_t j = 0; j < n; j++) {
      total_err -= A[i * n + j];
    }
  }
  if (abs(total_err) > 1e-9) {
    printf("(%s) vecmat first arg err: %f\n", application, total_err);
  }
}

int main() {
  unsigned long c_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long blas_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long enzyme_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long grad_loop_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long grad_blas_results[NUM_WARMUPS + NUM_RUNS];
  float *A = (float *)malloc(M * N * sizeof(float));
  float *x = (float *)malloc(M * sizeof(float));
  float *g = (float *)malloc(N * sizeof(float));

  random_init_2d(A, M, N);
  random_init(x, M);
  uniform_init(1.0f, g, N);

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dx = enzyme_vecmat_first(x, A, M, N);
    // {% endif %}
    gettimeofday(&stop, NULL);
    enzyme_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dx, A, M, N, "enzyme");
    free(dx);
    // {% endif %}
  }

  float *deadbeef = (float *)0xdeadbeef;
  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    F32Descriptor1D dx =
        grad_vecmat_first(deadbeef, x, 0, M, 1, deadbeef, A, 0, M, N, 1, 1);
    // {% endif %}
    gettimeofday(&stop, NULL);

    grad_loop_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dx.aligned, A, M, N, "grad_loops");
    free(dx.aligned);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    F32Descriptor1D dx = blas_grad_vecmat_first(deadbeef, x, 0, M, 1, deadbeef,
                                                A, 0, M, N, 1, 1);
    // {% endif %}
    gettimeofday(&stop, NULL);

    grad_blas_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dx.aligned, A, M, N, "grad_blas");
    free(dx.aligned);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dx = openblas_vecmat_first(x, A, g, M, N);
    // {% endif %}
    gettimeofday(&stop, NULL);
    blas_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dx, A, M, N, "handwritten_blas");
    free(dx);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dx = c_vecmat_first(x, A, g, M, N);
    // {% endif %}
    gettimeofday(&stop, NULL);
    c_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dx, A, M, N, "handwritten_c");
    free(dx);
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

  free(A);
  free(x);
  free(g);
}
