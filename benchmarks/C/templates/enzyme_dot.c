/**
 * Meant to be a playground to uncover the illegal updateAnalysis assertion
 * failures.
 */

extern void *malloc(unsigned long);
extern void free(void *);
extern float *__enzyme_autodiff(void *, ...);

void dot(float *a, float *b, float *c) {
  float *sum = (float *)malloc(sizeof(float));
  float val = a[0] + b[0];
  *sum = val;
  float final = *sum;
  // for (int i = 0; i < 4; i++) {
  //   *sum += a[i] * b[i];
  // }
  *c = final;
  free(sum);
}

int enzyme_dupnoneed;
float *ddot(float *a, float *da, float *b, float *db, float *c, float *dc) {
  return __enzyme_autodiff(dot, a, da, b, db, enzyme_dupnoneed, c, dc);
}

int main() {
  float a[4] = {1., 2., 3., 4.};
  float da[4] = {0};
  float b[4] = {3., 4., 5., 6.};
  float db[4] = {0};
  float c;
  float dc = 1.0;

  ddot(a, da, b, db, &c, &dc);
  for (int i = 0; i < 4; i++) {
    printf("%f", da[i]);
    if (i != 4 - 1) {
      printf(" ");
    }
  }
  printf("\n");
}
