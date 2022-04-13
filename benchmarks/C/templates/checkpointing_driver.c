#include "mlir_c_abi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUM_RUNS 10

extern double grad_pow_fully_cached(double, int64_t);
extern double grad_pow_smart_cached(double, int64_t);
extern double grad_pow_recomputed(double, int64_t);

unsigned long fc(double x, int64_t n) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double res = grad_pow_fully_cached(x, n);
  gettimeofday(&stop, NULL);
  double expected = (n * pow(x, n - 1));
  if (fabs(res - expected) > 1e-6) {
    printf("Incorrect fully-cached result: %f vs %f\n", res, expected);
  }
  return timediff(start, stop);
};

unsigned long sc(double x, int64_t n) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double res = grad_pow_smart_cached(x, n);
  gettimeofday(&stop, NULL);
  double expected = (n * pow(x, n - 1));
  if (fabs(res - expected) > 1e-6) {
    printf("Incorrect smart-cached result: %f vs %f\n", res, expected);
  }
  return timediff(start, stop);
};

unsigned long rc(double x, int64_t n) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  double res = grad_pow_recomputed(x, n);
  gettimeofday(&stop, NULL);
  if (fabs(res - (n * pow(x, n - 1))) > 1e-6) {
    printf("Incorrect result: %f\n", res);
  }
  return timediff(start, stop);
};

int main() {
  unsigned long results[NUM_RUNS];
  double x = 0.99999;
  // double x = 1.3;
  int64_t n = 8192;
  printf("Recomputed:\n");
  for (size_t run = 0; run < NUM_RUNS; run++) {
    results[run] = rc(x, n);
  }
  print_ul_arr(results, NUM_RUNS);

  printf("Smart cached:\n");
  for (size_t run = 0; run < NUM_RUNS; run++) {
    results[run] = sc(x, n);
  }
  print_ul_arr(results, NUM_RUNS);

  printf("Fully cached:\n");
  for (size_t run = 0; run < NUM_RUNS; run++) {
    results[run] = fc(x, n);
  }

  print_ul_arr(results, NUM_RUNS);
  return 0;
}
