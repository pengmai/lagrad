#include "mlir_c_abi.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#define NUM_WARMUPS {{ num_warmups }}
#define NUM_RUNS {{ num_runs }}
#define N {{ n }}

// {% if application == 'dot' %}
// {% if args == [0] or args == [1] %}
extern float *c_dot(float *, float *, int64_t);
extern float *openblas_dot(float *, float *, int64_t);
extern F32Descriptor1D blas_grad_dot(float *, float *, int64_t, int64_t,
                                     int64_t, float *, float *, int64_t,
                                     int64_t, int64_t);
extern F32Descriptor1D grad_dot(float *, float *, int64_t, int64_t, int64_t,
                                float *, float *, int64_t, int64_t, int64_t);
extern float *enzyme_dot(float *, float *, int64_t);
// {% elif args == [0, 1] %}
extern DotGradient grad_dot(float *, float *, int64_t, int64_t, int64_t,
                            float *, float *, int64_t, int64_t, int64_t);
// {% endif %}

void check_first_arg(float *da, float *b, int64_t size) {
  float error = 0.0;
  for (size_t i = 0; i < size; i++) {
    error += fabs(da[i] - b[i]);
  }
  if (error > 1e-9) {
    printf("Grad absolute total err (first arg): %f\n", error);
  }
}

void check_second_arg(F32Descriptor1D db, float *a) {
  float error = 0.0;
  for (size_t i = 0; i < db.size; i++) {
    error += fabs(db.aligned[i] - a[i]);
  }
  if (error > 1e-9) {
    printf("Grad absolute total err (second arg): %f\n", error);
  }
}

// {% endif %}

int main() {
  // {% if application == 'dot' %}
  unsigned long enzyme_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long grad_naive_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long grad_blas_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long c_results[NUM_WARMUPS + NUM_RUNS];
  unsigned long openblas_results[NUM_WARMUPS + NUM_RUNS];
  float *a = (float *)malloc(N * sizeof(float));
  float *b = (float *)malloc(N * sizeof(float));
  random_init(a, N);
  random_init(b, N);
  // {% endif %}

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    float *dot_enzyme = enzyme_dot(a, b, N);
    gettimeofday(&stop, NULL);

    enzyme_results[run] = timediff(start, stop);
    check_first_arg(dot_enzyme, b, N);
    free(dot_enzyme);
  }

  float *deadbeef = (float *)0xdeadbeef;
  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;

    gettimeofday(&start, NULL);
    // {% if application == 'dot' %}
    // {% if args == [0] or args == [1] %}
    F32Descriptor1D dot_grad =
        grad_dot(deadbeef, a, 0, N, 1, deadbeef, b, 0, N, 1);
    // {% elif args == [0, 1] %}
    DotGradient dot_grad = grad_dot(deadbeef, a, 0, N, 1, deadbeef, b, 0, N, 1);
    // {% endif %}
    // {% endif %}
    gettimeofday(&stop, NULL);

    grad_naive_results[run] = timediff(start, stop);
    // {% if application == 'dot' %}
    // {% if args == [0] %}
    check_first_arg(dot_grad.aligned, b, dot_grad.size);
    free(dot_grad.aligned);
    // {% elif args == [1] %}
    check_second_arg(dot_grad, a);
    free(dot_grad.aligned);
    // {% elif args == [0, 1] %}
    check_first_arg(dot_grad.da.aligned, b, dot_grad.da.size);
    check_second_arg(dot_grad.db.aligned, a, dot_grad.db.size);
    free(dot_grad.da.aligned);
    free(dot_grad.db.aligned);
    // {% endif %}
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    F32Descriptor1D dot_grad =
        blas_grad_dot(deadbeef, a, 0, N, 1, deadbeef, b, 0, N, 1);
    gettimeofday(&stop, NULL);

    grad_blas_results[run] = timediff(start, stop);
    check_first_arg(dot_grad.aligned, b, dot_grad.size);
    free(dot_grad.aligned);
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dot_c = c_dot(a, b, N);
    // {% endif %}
    gettimeofday(&stop, NULL);

    c_results[run] = timediff(start, stop);
    check_first_arg(dot_c, b, N);

    // {% if args == [0] %}
    free(dot_c);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dot_openblas = openblas_dot(a, b, N);
    // {% endif %}
    gettimeofday(&stop, NULL);

    openblas_results[run] = timediff(start, stop);
    check_first_arg(dot_openblas, b, N);

    free(dot_openblas);
  }

  printf("enzyme: ");
  print_ul_arr(enzyme_results + NUM_WARMUPS, NUM_RUNS);
  printf("grad_loops: ");
  print_ul_arr(grad_naive_results + NUM_WARMUPS, NUM_RUNS);
  printf("grad_blas: ");
  print_ul_arr(grad_blas_results + NUM_WARMUPS, NUM_RUNS);
  printf("handwritten_c: ");
  print_ul_arr(c_results + NUM_WARMUPS, NUM_RUNS);
  printf("handwritten_openblas: ");
  print_ul_arr(openblas_results + NUM_WARMUPS, NUM_RUNS);

  free(a);
  free(b);
}
