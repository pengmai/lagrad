#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
} F32Descriptor0D;

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} F32Descriptor1D;

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size_0;
  int64_t size_1;
  int64_t stride_0;
  int64_t stride_1;
} F32Descriptor2D;

typedef struct {
  F32Descriptor1D da;
  F32Descriptor1D db;
} DotGradient;

typedef struct {
  float *da;
  float *db;
} RawDotGradient;

typedef struct {
  F32Descriptor2D da;
  F32Descriptor2D db;
} MatVecGradient;

unsigned long timediff(struct timeval start, struct timeval stop);

void random_init(float *arr, size_t size);

void random_init_2d(float *arr, size_t m, size_t n);

void print_ul_arr(unsigned long *arr, size_t n);
