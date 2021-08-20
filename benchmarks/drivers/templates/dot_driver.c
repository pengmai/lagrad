#include "mlir_c_abi.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define NUM_WARMUPS {{ num_warmups }}
#define NUM_RUNS {{ num_runs }}
#define N {{ n }}

// {% if args == [0] or args == [1] %}
// extern float *c_dot(float*, float*, int64_t);
extern F32Descriptor1D grad_dot(float *, float *, int64_t, int64_t, int64_t,
                                float *, float *, int64_t, int64_t, int64_t);
// {% elif args == [0, 1] %}
extern DotGradient grad_dot(float *, float *, int64_t, int64_t, int64_t,
                            float *, float *, int64_t, int64_t, int64_t);
// {% endif %}

void check_first_arg(F32Descriptor1D da, float *b) {
  float error = 0.0;
  for (size_t i = 0; i < da.size; i++)
  {
    error += fabs(da.aligned[i] - b[i]);
  }
  if (error > 1e-9) {
    printf("Grad absolute total err (first arg): %f\n", error);
  }
}

void check_second_arg(F32Descriptor1D db, float *a) {
  float error = 0.0;
  for (size_t i = 0; i < db.size; i++)
  {
    error += fabs(db.aligned[i] - a[i]);
  }
  if (error > 1e-9) {
    printf("Grad absolute total err (second arg): %f\n", error);
  }
}

int main() {
  unsigned long grad_naive_results[NUM_WARMUPS + NUM_RUNS];
  float *a = (float *)malloc(N * sizeof(float));
  float *b = (float *)malloc(N * sizeof(float));
  random_init(a, N);
  random_init(b, N);

  float *deadbeef = (float *)0xdeadbeef;
  for (size_t run = 0; run < NUM_WARMUPS + NUM_RUNS; run++) {
    struct timeval start, stop;

    gettimeofday(&start, NULL);
    // {% if args == [0] or args == [1] %}
    F32Descriptor1D dot_grad =
        grad_dot(deadbeef, a, 0, N, 1, deadbeef, b, 0, N, 1);
    // {% elif args == [0, 1] %}
    DotGradient dot_grad = grad_dot(deadbeef, a, 0, N, 1, deadbeef, b, 0, N, 1);
    // {% endif %}
    gettimeofday(&stop, NULL);

    grad_naive_results[run] = timediff(start, stop);
    // {% if args == [0] %}
    check_first_arg(dot_grad, b);
    free(dot_grad.aligned);
    // {% elif args == [1] %}
    check_second_arg(dot_grad, a);
    free(dot_grad.aligned);
    // {% elif args == [0, 1] %}
    check_first_arg(dot_grad.da, b);
    check_second_arg(dot_grad.db, a);
    free(dot_grad.da.aligned);
    free(dot_grad.db.aligned);
    // {% endif %}
    /* code */
  }

  printf("grad_naive: ");
  print_ul_arr(grad_naive_results + NUM_WARMUPS, NUM_RUNS);

  free(a);
  free(b);
}
