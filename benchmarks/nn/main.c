#include "lagrad_utils.h"
#include "memusage.h"
#include "nn.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#define NUM_RUNS 6
#define CHECK_MEM false

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

MLPGrad enzyme_mlir_mlp_batched_adjoint(MLPModel *m, DataBatch *b) {
  return enzyme_mlir_mlp_batched(
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

float verify_array(int64_t size, float *actual, float *expected) {
  float err = 0;
  for (size_t i = 0; i < size; i++) {
    float item_err = fabs(actual[i] - expected[i]);
    if (item_err > err) {
      err = item_err;
    }
  }
  return err;
}

void verify_mlp_grad(MLPGrad *actual, MLPGrad *expected, const char *name) {
  float err;
  float tol = 3e-8;
  int64_t sizes[] = {actual->w0b.size_0 * actual->w0b.size_1, actual->b0b.size,
                     actual->w1b.size_0 * actual->w1b.size_1, actual->b1b.size,
                     actual->w2b.size_0 * actual->w2b.size_1, actual->b2b.size};
  float *all_actual[] = {actual->w0b.aligned, actual->b0b.aligned,
                         actual->w1b.aligned, actual->b1b.aligned,
                         actual->w2b.aligned, actual->b2b.aligned};
  float *all_expected[] = {expected->w0b.aligned, expected->b0b.aligned,
                           expected->w1b.aligned, expected->b1b.aligned,
                           expected->w2b.aligned, expected->b2b.aligned};
  const char *labels[] = {"weight 0", "bias 0",   "weight 1",
                          "bias 1",   "weight 2", "bias 2"};
  for (size_t i = 0; i < 6; i++) {
    err = verify_array(sizes[i], all_actual[i], all_expected[i]);
    if (err > tol) {
      printf("(%s) %s err: %.4e\n", name, labels[i], err);
    }
  }
}

unsigned long collect_mlp(MLPApp *app, MLPModel *model, DataBatch *batch,
                          MLPGrad *ref_grad) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  MLPGrad grad = app->func(model, batch);
  gettimeofday(&stop, NULL);
  // print_f_arr(grad.w0b.aligned, 10);
  // print_f_arr(grad.b2b.aligned, grad.b2b.size);
  // printf("Gradient of bias 1:\n");
  // print_f_arr(grad.b1b.aligned, 10);
  // check_mem_usage();
  if (CHECK_MEM) {
  } else {
    verify_mlp_grad(&grad, ref_grad, app->name);
  }

  free(grad.w0b.aligned);
  free(grad.b0b.aligned);
  free(grad.w1b.aligned);
  free(grad.b1b.aligned);
  free(grad.w2b.aligned);
  free(grad.b2b.aligned);
  return timediff(start, stop);
}

MLPGrad populate_ref(MLPModel *m, DataBatch *b) {
  return lagrad_mlp_batched_adjoint(m, b);
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

  MLPGrad ref_grad = populate_ref(&model, &transposed_batch);

  MLPApp apps[] = {
      // {.name = "LAGrad Nonbatched",
      //                 .batch = &batch,
      //                 .func = lagrad_mlp_adjoint},
      {.name = "LAGrad Batched",
       .batch = &transposed_batch,
       .func = lagrad_mlp_batched_adjoint},
      {.name = "Enzyme", .batch = &transposed_batch, .func = enzyme_mlp},
      {.name = "Enzyme/MLIR",
       .batch = &transposed_batch,
       .func = enzyme_mlir_mlp_batched_adjoint}};
  size_t num_apps = sizeof(apps) / sizeof(apps[0]);
  unsigned long results[NUM_RUNS];
  for (size_t app = 0; app < num_apps; app++) {
    printf("%s: ", apps[app].name);
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results[run] =
          collect_mlp(&apps[app], &model, apps[app].batch, &ref_grad);
    }
    print_ul_arr(results, NUM_RUNS);
  }
  free_mlp_model(&model);
  free(transposed_x);
}
