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
#define K {{ k }}

// #define NUM_WARMUPS 10
// #define NUM_RUNS 10
// #define M 128
// #define N 128
// #define K 128

// {% if args == [0] %}
extern float *c_matmul_first(float *, float *, float *, int64_t, int64_t,
                             int64_t);
extern float *openblas_matmul_first(float *, float *, float *, int64_t, int64_t,
                                    int64_t);
extern float *enzyme_matmul_first(float *, float *, int64_t, int64_t, int64_t);
extern F32Descriptor2D grad_matmul_first(/*A=*/float *, float *, int64_t,
                                         int64_t, int64_t, int64_t, int64_t,
                                         /*B=*/float *, float *, int64_t,
                                         int64_t, int64_t, int64_t, int64_t);
extern F32Descriptor2D blas_grad_matmul_first(
    /*A=*/float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*B=*/float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);
// {% endif %}

void check_first_arg(float *dA, float *B, size_t m, size_t n, size_t k,
                     const char *application) {
  float broadcasted[n];
  for (size_t i = 0; i < n; i++) {
    broadcasted[i] = 0.0f;
    for (size_t j = 0; j < k; j++) {
      broadcasted[i] += B[i * k + j];
    }
  }
  float error = 0.0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      error += abs(dA[i * n + j] - broadcasted[j]);
    }
  }
  if (error > 1e-9) {
    printf("(%s) matmul first arg error: %f\n", application, error);
  }
}

void check_second_arg() {}

int main() {
  unsigned long c_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long blas_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long enzyme_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long grad_loop_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long grad_blas_results[NUM_WARMUPS + NUM_RUNS];
  float *A = (float *)malloc(M * N * sizeof(float));
  float *B = (float *)malloc(N * K * sizeof(float));
  float *G = (float *)malloc(M * K * sizeof(float));
  random_init_2d(A, M, N);
  random_init_2d(B, N, K);
  uniform_init_2d(1.0, G, M, K);

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dA = enzyme_matmul_first(A, B, M, N, K);
    // {% endif %}
    gettimeofday(&stop, NULL);
    enzyme_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dA, B, M, N, K, "enzyme");
    free(dA);
    // {% endif %}
  }

  float *deadbeef = (float *)0xdeadbeef;
  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    F32Descriptor2D dA = grad_matmul_first(deadbeef, A, 0, M, N, 1, 1, deadbeef,
                                           B, 0, N, K, 1, 1);
    // {% endif %}
    gettimeofday(&stop, NULL);

    grad_loop_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dA.aligned, B, M, N, K, "grad_loops");
    free(dA.aligned);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    F32Descriptor2D dA = blas_grad_matmul_first(deadbeef, A, 0, M, N, 1, 1,
                                                deadbeef, B, 0, N, K, 1, 1);
    // {% endif %}
    gettimeofday(&stop, NULL);

    grad_blas_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dA.aligned, B, M, N, K, "grad_blas");
    free(dA.aligned);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dA = openblas_matmul_first(A, B, G, M, N, K);
    // {% endif %}
    gettimeofday(&stop, NULL);
    blas_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dA, B, M, N, K, "handwritten_blas");
    free(dA);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dA = c_matmul_first(A, B, G, M, N, K);
    // {% endif %}
    gettimeofday(&stop, NULL);
    c_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dA, B, M, N, K, "handwritten_c");
    free(dA);
    // {% endif %}
  }

  printf("enzyme: ");
  print_ul_arr(enzyme_results + NUM_WARMUPS, NUM_RUNS);
  printf("grad_loops: ");
  print_ul_arr(grad_loop_results + NUM_WARMUPS, NUM_RUNS);
  printf("grad_blas: ");
  print_ul_arr(grad_blas_results + NUM_WARMUPS, NUM_RUNS);
  printf("handwritten_c: ");
  print_ul_arr(c_results + NUM_WARMUPS, NUM_RUNS);
  printf("handwritten_blas: ");
  print_ul_arr(blas_results + NUM_WARMUPS, NUM_RUNS);
  free(A);
  free(B);
  free(G);
}
