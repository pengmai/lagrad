#include "nn.h"
#include <stdio.h>
#include <time.h>

#define NUM_RUNS 6

float *deadbeef = (float *)0xdeadbeef;

void print_f_arr(const float *arr, size_t n) {
  printf("[");
  for (size_t i = 0; i < n; i++) {
    printf("%.4e", arr[i]);
    if (i != n - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

float mlir_mlp_primal(MLPModel *m, DataBatch *b) {
  return mlir_mlp(deadbeef, b->features, 0, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE,
                  1, (int32_t *)deadbeef, b->labels, 0, BATCH_SIZE, 1, deadbeef,
                  m->weights0, 0, HIDDEN_SIZE, INPUT_SIZE, INPUT_SIZE, 1,
                  deadbeef, m->bias0, 0, HIDDEN_SIZE, 1, deadbeef, m->weights1,
                  0, HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, 1, deadbeef,
                  m->bias1, 0, HIDDEN_SIZE, 1, deadbeef, m->weights2, 0,
                  OUTPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, 1, deadbeef, m->bias2,
                  0, OUTPUT_SIZE, 1);
}

MLPGrad lagrad_mlp_adjoint(MLPModel *m, DataBatch *b) {
  return lagrad_mlp(
      deadbeef, b->features, 0, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 1,
      (int32_t *)deadbeef, b->labels, 0, BATCH_SIZE, 1, deadbeef, m->weights0,
      0, HIDDEN_SIZE, INPUT_SIZE, INPUT_SIZE, 1, deadbeef, m->bias0, 0,
      HIDDEN_SIZE, 1, deadbeef, m->weights1, 0, HIDDEN_SIZE, HIDDEN_SIZE,
      HIDDEN_SIZE, 1, deadbeef, m->bias1, 0, HIDDEN_SIZE, 1, deadbeef,
      m->weights2, 0, OUTPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, 1, deadbeef,
      m->bias2, 0, OUTPUT_SIZE, 1);
}

unsigned long collect_lagrad_mlp(MLPModel *model, DataBatch *batch) {
  clock_t start, stop;
  start = clock();
  MLPGrad grad = lagrad_mlp_adjoint(model, batch);
  stop = clock();
  return stop - start;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: %s <model file> <data_file>\n", argv[0]);
    return 1;
  }
  MLPModel model = read_mlp_model(argv[1]);
  DataBatch batch = read_data_batch(argv[2]);
  clock_t start, stop;
  printf("LAGrad:\n");
  for (size_t run = 0; run < NUM_RUNS; run++) {
    start = clock();
    MLPGrad grad = lagrad_mlp_adjoint(&model, &batch);
    stop = clock();
    printf("elapsed: %lu\n", stop - start);
    free(grad.w0b.aligned);
    free(grad.b0b.aligned);
    free(grad.w1b.aligned);
    free(grad.b1b.aligned);
    free(grad.w2b.aligned);
    free(grad.b2b.aligned);
  }
  // print_f_arr(grad.w2b.aligned, 10);
  // printf("MLIR Loss : %f\n", mlir_mlp_primal(&model, &batch));
  // enzyme_primal(&model, &batch);

  printf("Enzyme:\n");

  float *w0b = calloc(INPUT_SIZE * HIDDEN_SIZE, sizeof(float));
  float *b0b = calloc(HIDDEN_SIZE, sizeof(float));
  float *w1b = calloc(HIDDEN_SIZE * HIDDEN_SIZE, sizeof(float));
  float *b1b = calloc(HIDDEN_SIZE, sizeof(float));
  float *w2b = calloc(HIDDEN_SIZE * OUTPUT_SIZE, sizeof(float));
  float *b2b = calloc(OUTPUT_SIZE, sizeof(float));
  for (size_t run = 0; run < NUM_RUNS; run++) {

    start = clock();
    enzyme_mlp(&model, &batch, w0b, b0b, w1b, b1b, w2b, b2b);
    stop = clock();
    printf("elapsed: %lu\n", stop - start);
  }
  free(w0b);
  free(b0b);
  free(w1b);
  free(b1b);
  free(w2b);
  free(b2b);
  free_mlp_model(&model);
}
