#pragma once
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

void enzyme_primal(MLPModel *m, DataBatch *b);
void enzyme_mlp(MLPModel *m, DataBatch *b, float *w0b, float *b0b, float *w1b,
                float *b1b, float *w2b, float *b2b);

typedef struct F32Descriptor1D {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} F32Descriptor1D;

typedef struct D32Descriptor2D {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size_0;
  int64_t size_1;
  int64_t stride_0;
  int64_t stride_1;
} F32Descriptor2D;

typedef struct MLPGrad {
  F32Descriptor2D w0b;
  F32Descriptor1D b0b;
  F32Descriptor2D w1b;
  F32Descriptor1D b1b;
  F32Descriptor2D w2b;
  F32Descriptor1D b2b;
} MLPGrad;

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
// func @lagrad_mlp(%arg0: tensor<64x784xf32>, %arg1: tensor<64xi32>, %arg2:
// tensor<512x784xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512x512xf32>,
// %arg5: tensor<512xf32>, %arg6: tensor<10x512xf32>, %arg7: tensor<10xf32>) ->
// (tensor<512x784xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>,
// tensor<10x512xf32>, tensor<10xf32>) {