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

void nn_softmax(int length, const float *activations, float *outp) {
  int i;
  float sum, max;

  for (i = 1, max = activations[0]; i < length; i++) {
    if (activations[i] > max) {
      max = activations[i];
    }
  }

  for (i = 0, sum = 0; i < length; i++) {
    sum += exp(activations[i] - max);
  }

  for (i = 0; i < length; i++) {
    outp[i] = exp(activations[i] - max) / sum;
  }
}

float enzyme_nn_hypothesis(const float *input, const int32_t *labels,
                           const float *w0, const float *b0, const float *w1,
                           const float *b1, const float *w2, const float *b2) {
  float *hidden0 = malloc(HIDDEN_SIZE * sizeof(float));
  float *hidden1 = malloc(HIDDEN_SIZE * sizeof(float));
  float activations[OUTPUT_SIZE];
  float output[OUTPUT_SIZE];
  float loss = 0;
  for (int b = 0; b < BATCH_SIZE; b++) {
    memcpy(hidden0, b0, HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
      for (int j = 0; j < INPUT_SIZE; j++) {
        hidden0[i] += w0[i * INPUT_SIZE + j] * input[b * INPUT_SIZE + j];
      }
    }
    relu(HIDDEN_SIZE, hidden0);

    memcpy(hidden1, b1, HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
      for (int j = 0; j < HIDDEN_SIZE; j++) {
        hidden1[i] += w1[i * HIDDEN_SIZE + j] * hidden0[j];
      }
    }
    relu(HIDDEN_SIZE, hidden1);

    memcpy(activations, b2, OUTPUT_SIZE * sizeof(float));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      for (int j = 0; j < HIDDEN_SIZE; j++) {
        activations[i] += w2[i * HIDDEN_SIZE + j] * hidden1[j];
      }
    }

    nn_softmax(OUTPUT_SIZE, activations, output);
    loss -= log(output[labels[b]]);
  }
  free(hidden0);
  free(hidden1);
  // printf("avg loss: %f\n", loss / BATCH_SIZE);
  return loss;
}

// float enzyme_nn_loss(const float *input, const int32_t *labels, const float
// *w0,
//                      const float *b0, const float *w1, const float *b1,
//                      const float *w2, const float *b2) {
//   float loss = 0.0f;

// }

extern int enzyme_const;
extern void __enzyme_autodiff(void *, ...);

void enzyme_primal(MLPModel *m, DataBatch *b) {
  enzyme_nn_hypothesis(b->features, b->labels, m->weights0, m->bias0,
                       m->weights1, m->bias1, m->weights2, m->bias2);
}

void enzyme_mlp(MLPModel *m, DataBatch *b, float *w0b, float *b0b, float *w1b,
                float *b1b, float *w2b, float *b2b) {
  __enzyme_autodiff(enzyme_nn_hypothesis, enzyme_const, b->features, b->labels,
                    m->weights0, w0b, m->bias0, b0b, m->weights1, w1b, m->bias1,
                    b1b, m->weights2, w2b, m->bias2, b2b);
}
