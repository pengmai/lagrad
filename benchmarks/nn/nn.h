#pragma once
#include "lagrad_utils.h"
#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 512
#define OUTPUT_SIZE 10
#define BATCH_SIZE 64

typedef struct MLPModel {
  float *weights0, *bias0, *weights1, *bias1, *weights2, *bias2;
} MLPModel;

typedef struct DataBatch {
  float *features;
  int32_t *labels;
} DataBatch;

MLPModel read_mlp_model(const char *filename);
DataBatch read_data_batch(const char *filename);

void free_mlp_model(MLPModel *m);
void free_data_batch(DataBatch *b);

typedef struct MLPGrad {
  F32Descriptor2D w0b;
  F32Descriptor1D b0b;
  F32Descriptor2D w1b;
  F32Descriptor1D b1b;
  F32Descriptor2D w2b;
  F32Descriptor1D b2b;
} MLPGrad;

void enzyme_primal(MLPModel *m, DataBatch *b);
MLPGrad enzyme_mlp(MLPModel *m, DataBatch *b);

float mlir_mlp(/*input=*/float *, float *input, int64_t, int64_t, int64_t,
               int64_t, int64_t,
               /*labels=*/int32_t *, int32_t *labels, int64_t, int64_t, int64_t,
               /*w0=*/float *, float *w0, int64_t, int64_t, int64_t, int64_t,
               int64_t,
               /*b0=*/float *, float *b0, int64_t, int64_t, int64_t,
               /*w1=*/float *, float *w1, int64_t, int64_t, int64_t, int64_t,
               int64_t,
               /*b1=*/float *, float *b1, int64_t, int64_t, int64_t,
               /*w2=*/float *, float *w2, int64_t, int64_t, int64_t, int64_t,
               int64_t,
               /*b2=*/float *, float *b2, int64_t, int64_t, int64_t);

float mlir_mlp_batched(
    /*input=*/float *, float *input, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*labels=*/int32_t *, int32_t *labels, int64_t, int64_t, int64_t,
    /*w0=*/float *, float *w0, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*b0=*/float *, float *b0, int64_t, int64_t, int64_t,
    /*w1=*/float *, float *w1, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*b1=*/float *, float *b1, int64_t, int64_t, int64_t,
    /*w2=*/float *, float *w2, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*b2=*/float *, float *b2, int64_t, int64_t, int64_t);

MLPGrad lagrad_mlp(
    /*input=*/float *, float *input, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*labels=*/int32_t *, int32_t *labels, int64_t, int64_t, int64_t,
    /*w0=*/float *, float *w0, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*b0=*/float *, float *b0, int64_t, int64_t, int64_t,
    /*w1=*/float *, float *w1, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*b1=*/float *, float *b1, int64_t, int64_t, int64_t,
    /*w2=*/float *, float *w2, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*b2=*/float *, float *b2, int64_t, int64_t, int64_t);

MLPGrad lagrad_mlp_batched(
    /*input=*/float *, float *input, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*labels=*/int32_t *, int32_t *labels, int64_t, int64_t, int64_t,
    /*w0=*/float *, float *w0, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*b0=*/float *, float *b0, int64_t, int64_t, int64_t,
    /*w1=*/float *, float *w1, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*b1=*/float *, float *b1, int64_t, int64_t, int64_t,
    /*w2=*/float *, float *w2, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*b2=*/float *, float *b2, int64_t, int64_t, int64_t);
