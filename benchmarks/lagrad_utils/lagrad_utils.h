#pragma once
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
} F64Descriptor0D;

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
  double *allocated;
  double *aligned;
  int64_t offset;
  int64_t size_0;
  int64_t size_1;
  int64_t size_2;
  int64_t size_3;
  int64_t stride_0;
  int64_t stride_1;
  int64_t stride_2;
  int64_t stride_3;
} F64Descriptor4D;

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size_0;
  int64_t size_1;
  int64_t stride_0;
  int64_t stride_1;
} F32Descriptor2D;

unsigned long timediff(struct timeval start, struct timeval stop);

void print_ul_arr(unsigned long *arr, size_t n);

void print_f_arr(float *arr, size_t n);

void print_d_arr(const double *arr, size_t n);

void print_f_arr_2d(float *arr, size_t m, size_t n);

void print_d_arr_2d(double *arr, size_t m, size_t n);

void print_d_arr_3d(double *arr, size_t m, size_t n, size_t k);
