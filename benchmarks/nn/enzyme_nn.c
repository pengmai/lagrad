#include "nn.h"
#include <math.h>
#include <string.h>

void relu(int size, float *x) {
  for (int i = 0; i < size; i++) {
    if (x[i] < 0.0) {
      x[i] = 0.0;
    }
  }
}

void matmult(int M, int N, int K, const float *restrict A,
             const float *restrict B, float *restrict C) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// h: [M, N]
// b: [M]
void broadcast_add(int M, int N, float *h, const float *b) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      h[i * N + j] += b[i];
    }
  }
}

float nn_crossentropy(const int b, const int length, const float *activations,
                      const int32_t *labels) {
  float maxs[b];
  float sums[b];
  for (int i = 0; i < b; i++) {
    maxs[i] = activations[i];
    sums[i] = 0;
  }

  for (int i = 0; i < length; i++) {
    for (int j = 0; j < b; j++) {
      maxs[j] =
          activations[i * b + j] > maxs[j] ? activations[i * b + j] : maxs[j];
    }
  }

  for (int i = 0; i < length; i++) {
    for (int j = 0; j < b; j++) {
      sums[j] += exp(activations[i * b + j] - maxs[j]);
    }
  }

  float sum_output = 0;
  for (int j = 0; j < b; j++) {
    sum_output += -log(exp(activations[labels[j] * b + j] - maxs[j]) / sums[j]);
  }
  return sum_output / b;
}

float ec_mlp_batched(const float *restrict input,
                     const int32_t *restrict labels, const float *restrict w0,
                     const float *restrict b0, const float *restrict w1,
                     const float *restrict b1, const float *restrict w2,
                     const float *restrict b2) {
  float *hidden0 = malloc(HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
  float *hidden1 = malloc(HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
  float *activations = malloc(OUTPUT_SIZE * BATCH_SIZE * sizeof(float));
  memset(hidden0, 0, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
  memset(hidden1, 0, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
  memset(activations, 0, OUTPUT_SIZE * BATCH_SIZE * sizeof(float));

  // first layer
  matmult(HIDDEN_SIZE, BATCH_SIZE, INPUT_SIZE, w0, input, hidden0);
  broadcast_add(HIDDEN_SIZE, BATCH_SIZE, hidden0, b0);
  relu(HIDDEN_SIZE * BATCH_SIZE, hidden0);

  // second layer
  matmult(HIDDEN_SIZE, BATCH_SIZE, HIDDEN_SIZE, w1, hidden0, hidden1);
  broadcast_add(HIDDEN_SIZE, BATCH_SIZE, hidden1, b1);
  relu(HIDDEN_SIZE * BATCH_SIZE, hidden1);

  // output layer
  matmult(OUTPUT_SIZE, BATCH_SIZE, HIDDEN_SIZE, w2, hidden1, activations);
  broadcast_add(OUTPUT_SIZE, BATCH_SIZE, activations, b2);
  float loss = nn_crossentropy(BATCH_SIZE, OUTPUT_SIZE, activations, labels);

  free(hidden0);
  free(hidden1);
  free(activations);
  return loss;
}

extern int enzyme_const;
extern void __enzyme_autodiff(void *, ...);

float enzyme_primal(MLPModel *m, DataBatch *b) {
  return ec_mlp_batched(b->features, b->labels, m->weights0, m->bias0,
                        m->weights1, m->bias1, m->weights2, m->bias2);
}

MLPGrad enzyme_mlp(MLPModel *m, DataBatch *b) {
  float *w0b = calloc(INPUT_SIZE * HIDDEN_SIZE, sizeof(float));
  float *b0b = calloc(HIDDEN_SIZE, sizeof(float));
  float *w1b = calloc(HIDDEN_SIZE * HIDDEN_SIZE, sizeof(float));
  float *b1b = calloc(HIDDEN_SIZE, sizeof(float));
  float *w2b = calloc(HIDDEN_SIZE * OUTPUT_SIZE, sizeof(float));
  float *b2b = calloc(OUTPUT_SIZE, sizeof(float));
  MLPGrad grad = {.w0b = {.allocated = NULL,
                          .aligned = w0b,
                          .offset = 0,
                          .size_0 = INPUT_SIZE,
                          .size_1 = HIDDEN_SIZE,
                          .stride_0 = HIDDEN_SIZE,
                          .stride_1 = 1},
                  .b0b = {.allocated = NULL,
                          .aligned = b0b,
                          .offset = 0,
                          .size = HIDDEN_SIZE,
                          .stride = 1},
                  .w1b = {.allocated = NULL,
                          .aligned = w1b,
                          .offset = 0,
                          .size_0 = HIDDEN_SIZE,
                          .size_1 = HIDDEN_SIZE,
                          .stride_0 = HIDDEN_SIZE,
                          .stride_1 = 1},
                  .b1b = {.allocated = NULL,
                          .aligned = b1b,
                          .offset = 0,
                          .size = HIDDEN_SIZE,
                          .stride = 1},
                  .w2b = {.allocated = NULL,
                          .aligned = w2b,
                          .offset = 0,
                          .size_0 = HIDDEN_SIZE,
                          .size_1 = OUTPUT_SIZE,
                          .stride_0 = OUTPUT_SIZE,
                          .stride_1 = 1},
                  .b2b = {.allocated = NULL,
                          .aligned = b2b,
                          .offset = 0,
                          .size = OUTPUT_SIZE,
                          .stride = 1}};
  __enzyme_autodiff(ec_mlp_batched, enzyme_const, b->features, b->labels,
                    m->weights0, w0b, m->bias0, b0b, m->weights1, w1b, m->bias1,
                    b1b, m->weights2, w2b, m->bias2, b2b);
  return grad;
}
