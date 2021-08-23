#include "mlir_c_abi.h"
#include "shared_types.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#define NUM_WARMUPS {{ num_warmups }}
#define NUM_RUNS {{ num_runs }}
#define N {{ n }}

// #define NUM_WARMUPS 10
// #define NUM_RUNS 10
// #define N 131072

// {% if args == [0] %}
extern float *c_dot_first(float *, float *, float, int64_t);
extern float *openblas_dot_first(float *, float *, int64_t);
extern F32Descriptor1D blas_grad_dot_first(float *, float *, int64_t, int64_t,
                                           int64_t, float *, float *, int64_t,
                                           int64_t, int64_t);
extern F32Descriptor1D grad_dot_first(float *, float *, int64_t, int64_t,
                                      int64_t, float *, float *, int64_t,
                                      int64_t, int64_t);
extern float *enzyme_dot_first(float *, float *, int64_t);
// {% elif args == [1] %}
extern float *c_dot_second(float *, float *, float, int64_t);
extern float *openblas_dot_second(float *, float *, int64_t);
extern F32Descriptor1D blas_grad_dot_second(float *, float *, int64_t, int64_t,
                                            int64_t, float *, float *, int64_t,
                                            int64_t, int64_t);
extern F32Descriptor1D grad_dot_second(float *, float *, int64_t, int64_t,
                                       int64_t, float *, float *, int64_t,
                                       int64_t, int64_t);
extern float *enzyme_dot_second(float *, float *, int64_t);
// {% elif args == [0, 1] %}
extern RawDotGradient c_dot_both(float *a, float *b, float g, int64_t size);
extern RawDotGradient openblas_dot_both(float *a, float *b, int size);
extern DotGradient blas_grad_dot_both(float *, float *, int64_t, int64_t,
                                      int64_t, float *, float *, int64_t,
                                      int64_t, int64_t);
extern DotGradient grad_dot_both(float *, float *, int64_t, int64_t, int64_t,
                                 float *, float *, int64_t, int64_t, int64_t);
extern RawDotGradient enzyme_dot_both(float *, float *, int64_t);
// {% endif %}

void check_first_arg(float *da, float *b, int64_t size, const char *app) {
  float error = 0.0;
  for (size_t i = 0; i < size; i++) {
    error += fabs(da[i] - b[i]);
  }
  if (error > 1e-9) {
    printf("(%s) Absolute total err (first arg): %f\n", app, error);
  }
}

void check_second_arg(float *db, float *a, int64_t size, const char *app) {
  float error = 0.0;
  for (size_t i = 0; i < size; i++) {
    error += fabs(db[i] - a[i]);
  }
  if (error > 1e-9) {
    printf("(%s) Absolute total err (second arg): %f\n", app, error);
  }
}

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
    // {% if args == [0] %}
    float *dot_enzyme_first = enzyme_dot_first(a, b, N);
    // {% elif args == [1] %}
    float *dot_enzyme_second = enzyme_dot_second(a, b, N);
    // {% else %}
    RawDotGradient dot_enzyme_both = enzyme_dot_both(a, b, N);
    // {% endif %}
    gettimeofday(&stop, NULL);

    enzyme_results[run] = timediff(start, stop);
    // {% if args == [0] %}
    check_first_arg(dot_enzyme_first, b, N, "enzyme");
    free(dot_enzyme_first);
    // {% elif args == [1] %}
    check_second_arg(dot_enzyme_second, a, N, "enzyme");
    free(dot_enzyme_second);
    // {% else %}
    check_first_arg(dot_enzyme_both.da, b, N, "enzyme");
    check_second_arg(dot_enzyme_both.db, a, N, "enzyme");
    free(dot_enzyme_both.da);
    free(dot_enzyme_both.db);
    // {% endif%}
  }

  float *deadbeef = (float *)0xdeadbeef;
  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;

    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    F32Descriptor1D dot_grad_first =
        grad_dot_first(deadbeef, a, 0, N, 1, deadbeef, b, 0, N, 1);
    // {% elif args == [1] %}
    F32Descriptor1D dot_grad_second =
        grad_dot_second(deadbeef, a, 0, N, 1, deadbeef, b, 0, N, 1);
    // {% else %}
    DotGradient dot_grad =
        grad_dot_both(deadbeef, a, 0, N, 1, deadbeef, b, 0, N, 1);
    // {% endif %}
    gettimeofday(&stop, NULL);

    grad_naive_results[run] = timediff(start, stop);
    // {% if args == [0] %}
    check_first_arg(dot_grad_first.aligned, b, dot_grad_first.size,
                    "grad_loop");
    free(dot_grad_first.aligned);
    // {% elif args == [1] %}
    check_second_arg(dot_grad_second.aligned, a, dot_grad_second.stride,
                     "grad_loop");
    free(dot_grad_second.aligned);
    // {% else %}
    check_first_arg(dot_grad.da.aligned, b, dot_grad.da.size, "grad_loop");
    check_second_arg(dot_grad.db.aligned, a, dot_grad.db.size, "grad_loop");
    free(dot_grad.da.aligned);
    free(dot_grad.db.aligned);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    F32Descriptor1D dot_grad_first =
        blas_grad_dot_first(deadbeef, a, 0, N, 1, deadbeef, b, 0, N, 1);
    // {% elif args == [1] %}
    F32Descriptor1D dot_grad_second =
        blas_grad_dot_second(deadbeef, a, 0, N, 1, deadbeef, b, 0, N, 1);
    // {% else %}
    DotGradient dot_grad_both =
        blas_grad_dot_both(deadbeef, a, 0, N, 1, deadbeef, b, 0, N, 1);
    // {% endif %}
    gettimeofday(&stop, NULL);

    grad_blas_results[run] = timediff(start, stop);
    // {% if args == [0] %}
    check_first_arg(dot_grad_first.aligned, b, dot_grad_first.size,
                    "grad_blas");
    free(dot_grad_first.aligned);
    // {% elif args == [1] %}
    check_second_arg(dot_grad_second.aligned, a, dot_grad_second.size,
                     "grad_blas");
    free(dot_grad_second.aligned);
    // {% else %}
    check_first_arg(dot_grad_both.da.aligned, b, dot_grad_both.da.size,
                    "grad_blas");
    check_second_arg(dot_grad_both.db.aligned, a, dot_grad_both.db.size,
                     "grad_blas");
    free(dot_grad_both.da.aligned);
    free(dot_grad_both.db.aligned);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dot_c_first = c_dot_first(a, b, 1.0f, N);
    // {% elif args == [1] %}
    float *dot_c_second = c_dot_second(a, b, 1.0f, N);
    // {% else %}
    RawDotGradient dot_c_both = c_dot_both(a, b, 1.0f, N);
    // {% endif %}
    gettimeofday(&stop, NULL);

    c_results[run] = timediff(start, stop);
    // {% if args == [0] %}
    check_first_arg(dot_c_first, b, N, "handwritten_c");
    free(dot_c_first);
    // {% elif args == [1] %}
    check_second_arg(dot_c_second, a, N, "handwritten_c");
    free(dot_c_second);
    // {% else %}
    check_first_arg(dot_c_both.da, b, N, "handwritten_c");
    check_second_arg(dot_c_both.db, a, N, "handwritten_c");
    free(dot_c_both.da);
    free(dot_c_both.db);
    // {% endif %}
  }

  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    // {% if args == [0] %}
    float *dot_openblas_first = openblas_dot_first(a, b, N);
    // {% elif args == [1] %}
    float *dot_openblas_second = openblas_dot_second(a, b, N);
    // {% else %}
    RawDotGradient dot_openblas_both = openblas_dot_both(a, b, N);
    // {% endif %}
    gettimeofday(&stop, NULL);

    openblas_results[run] = timediff(start, stop);

    // {% if args == [0] %}
    check_first_arg(dot_openblas_first, b, N, "handwritten_blas");
    free(dot_openblas_first);
    // {% elif args == [1] %}
    check_second_arg(dot_openblas_second, a, N, "handwritten_blas");
    free(dot_openblas_second);
    // {% else %}
    check_first_arg(dot_openblas_both.da, b, N, "handwritten_blas");
    check_second_arg(dot_openblas_both.db, a, N, "handwritten_blas");
    free(dot_openblas_both.da);
    free(dot_openblas_both.db);
    // {% endif %}
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
