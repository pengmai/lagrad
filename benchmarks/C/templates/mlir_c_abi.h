#pragma once

#include <stdint.h>
// #include <stdlib.h>
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
  double *allocated;
  double *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} F64Descriptor1D;

typedef struct {
  double *allocated;
  double *aligned;
  int64_t offset;
  int64_t size_0;
  int64_t size_1;
  int64_t stride_0;
  int64_t stride_1;
} F64Descriptor2D;

typedef struct {
  double *allocated;
  double *aligned;
  int64_t offset;
  int64_t size_0;
  int64_t size_1;
  int64_t size_2;
  int64_t stride_0;
  int64_t stride_1;
  int64_t stride_2;
} F64Descriptor3D;

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
  F32Descriptor2D da;
  F32Descriptor2D db;
} MatVecGradient;

unsigned long timediff(struct timeval start, struct timeval stop);

void init_range(double *arr, size_t size);

void random_init(float *arr, size_t size);

void random_init_2d(float *arr, size_t m, size_t n);

void random_init_d_2d(double *arr, size_t m, size_t n);

void uniform_init(float val, float *arr, size_t size);

void uniform_init_d(double val, double *arr, size_t size);

void uniform_init_2d(float val, float *arr, size_t m, size_t n);

void print_ul_arr(unsigned long *arr, size_t n);

void print_f_arr(float *arr, size_t n);

void print_d_arr(const double *arr, size_t n);

void print_f_arr_2d(float *arr, size_t m, size_t n);

void print_d_arr_2d(double *arr, size_t m, size_t n);

void print_d_arr_3d(double *arr, size_t m, size_t n, size_t k);
