#include "lagrad_utils.h"
#include "memusage.h"
#include "nn.h"
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#define NUM_RUNS 6

float *deadbeef = (float *)0xdeadbeef;
RunProcDyn rpd;
void check_mem_usage() {
  run_get_dynamic_proc_info(getpid(), &rpd);
  printf("%zu\t%zu\n", rpd.rss, rpd.vsize);
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

float mlir_mlp_primal_batched(MLPModel *m, DataBatch *b) {
  return mlir_mlp_batched(
      deadbeef, b->features, 0, INPUT_SIZE, BATCH_SIZE, BATCH_SIZE, 1,
      (int32_t *)deadbeef, b->labels, 0, BATCH_SIZE, 1, deadbeef, m->weights0,
      0, HIDDEN_SIZE, INPUT_SIZE, INPUT_SIZE, 1, deadbeef, m->bias0, 0,
      HIDDEN_SIZE, 1, deadbeef, m->weights1, 0, HIDDEN_SIZE, HIDDEN_SIZE,
      HIDDEN_SIZE, 1, deadbeef, m->bias1, 0, HIDDEN_SIZE, 1, deadbeef,
      m->weights2, 0, OUTPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, 1, deadbeef,
      m->bias2, 0, OUTPUT_SIZE, 1);
}

MLPGrad lagrad_mlp_batched_adjoint(MLPModel *m, DataBatch *b) {
  return lagrad_mlp_batched(
      deadbeef, b->features, 0, INPUT_SIZE, BATCH_SIZE, BATCH_SIZE, 1,
      (int32_t *)deadbeef, b->labels, 0, BATCH_SIZE, 1, deadbeef, m->weights0,
      0, HIDDEN_SIZE, INPUT_SIZE, INPUT_SIZE, 1, deadbeef, m->bias0, 0,
      HIDDEN_SIZE, 1, deadbeef, m->weights1, 0, HIDDEN_SIZE, HIDDEN_SIZE,
      HIDDEN_SIZE, 1, deadbeef, m->bias1, 0, HIDDEN_SIZE, 1, deadbeef,
      m->weights2, 0, OUTPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, 1, deadbeef,
      m->bias2, 0, OUTPUT_SIZE, 1);
}

typedef struct MLPApp {
  const char *name;
  DataBatch *batch;
  MLPGrad (*func)(MLPModel *model, DataBatch *batch);
} MLPApp;

unsigned long collect_mlp(MLPApp *app, MLPModel *model, DataBatch *batch) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  MLPGrad grad = app->func(model, batch);
  gettimeofday(&stop, NULL);
  // print_f_arr(grad.w0b.aligned, 10);
  // print_f_arr(grad.b2b.aligned, grad.b2b.size);
  // printf("Gradient of bias 1:\n");
  // print_f_arr(grad.b1b.aligned, 30);
  // check_mem_usage();

  free(grad.w0b.aligned);
  free(grad.b0b.aligned);
  free(grad.w1b.aligned);
  free(grad.b1b.aligned);
  free(grad.w2b.aligned);
  free(grad.b2b.aligned);
  return timediff(start, stop);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: %s <model file> <data_file>\n", argv[0]);
    return 1;
  }
  MLPModel model = read_mlp_model(argv[1]);
  float *transposed_x = malloc(INPUT_SIZE * BATCH_SIZE * sizeof(float));
  DataBatch batch = read_data_batch(argv[2]);
  DataBatch transposed_batch = batch;
  transposed_batch.features = transposed_x;
  for (size_t i = 0; i < BATCH_SIZE; i++) {
    for (size_t j = 0; j < INPUT_SIZE; j++) {
      transposed_x[j * BATCH_SIZE + i] = batch.features[i * INPUT_SIZE + j];
    }
  }

  MLPApp apps[] = {// {.name = "LAGrad Nonbatched",
                   //  .batch = &batch,
                   //  .func = collect_lagrad_mlp},
                   {.name = "LAGrad Batched",
                    .batch = &transposed_batch,
                    .func = lagrad_mlp_batched_adjoint},
                   {.name = "Enzyme", .batch = &batch, .func = enzyme_mlp}};
  size_t num_apps = sizeof(apps) / sizeof(apps[0]);
  unsigned long results[NUM_RUNS];
  for (size_t app = 0; app < num_apps; app++) {
    printf("%s: ", apps[app].name);
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results[run] = collect_mlp(&apps[app], &model, apps[app].batch);
    }
    print_ul_arr(results, NUM_RUNS);
  }
  free_mlp_model(&model);
  free(transposed_x);
}
