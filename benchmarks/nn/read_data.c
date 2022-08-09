#include "nn.h"

MLPModel read_mlp_model(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file \"%s\"\n", filename);
    exit(EXIT_FAILURE);
  }
  float *weights0 = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
  float *bias0 = malloc(HIDDEN_SIZE * sizeof(float));
  float *weights1 = malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
  float *bias1 = malloc(HIDDEN_SIZE * sizeof(float));
  float *weights2 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
  float *bias2 = malloc(OUTPUT_SIZE * sizeof(float));
  for (size_t i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
    fscanf(fp, "%f", &weights0[i]);
  }
  for (size_t i = 0; i < HIDDEN_SIZE; i++) {
    fscanf(fp, "%f", &bias0[i]);
  }
  for (size_t i = 0; i < HIDDEN_SIZE * HIDDEN_SIZE; i++) {
    fscanf(fp, "%f", &weights1[i]);
  }
  for (size_t i = 0; i < HIDDEN_SIZE; i++) {
    fscanf(fp, "%f", &bias1[i]);
  }
  for (size_t i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
    fscanf(fp, "%f", &weights2[i]);
  }
  for (size_t i = 0; i < OUTPUT_SIZE; i++) {
    fscanf(fp, "%f", &bias2[i]);
  }

  fclose(fp);
  MLPModel m = {.weights0 = weights0,
                .bias0 = bias0,
                .weights1 = weights1,
                .bias1 = bias1,
                .weights2 = weights2,
                .bias2 = bias2};
  return m;
}

DataBatch read_data_batch(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file \"%s\"\n", filename);
    exit(EXIT_FAILURE);
  }

  float *features = malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
  int32_t *labels = malloc(BATCH_SIZE * sizeof(int32_t));
  for (size_t i = 0; i < BATCH_SIZE * INPUT_SIZE; i++) {
    fscanf(fp, "%f", &features[i]);
  }
  for (size_t i = 0; i < BATCH_SIZE; i++) {
    fscanf(fp, "%d", &labels[i]);
  }

  fclose(fp);
  DataBatch b = {.features = features, .labels = labels};
  return b;
}

void free_mlp_model(MLPModel *m) {
  free(m->weights0);
  free(m->bias0);
  free(m->weights1);
  free(m->bias1);
  free(m->weights2);
  free(m->bias2);
}

void free_data_batch(DataBatch *b) {
  free(b->features);
  free(b->labels);
}
