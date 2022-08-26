#pragma once
#include "lagrad_utils.h"
#include "stdbool.h"

typedef struct Matrix {
  int nrows;
  int ncols;
  double *data;
} Matrix;

typedef struct Triangle {
  int verts[3];
} Triangle;

typedef struct HandModel {
  int n_bones, n_vertices, n_triangles;
  const char **bone_names;
  int *parents;
  double *base_relatives, *inverse_base_absolutes, *base_positions, *weights;
  int *triangles;
  bool is_mirrored;
} HandModel;

typedef struct HandInput {
  HandModel model;
  int *correspondences;
  int n_theta, n_pts;
  double *points, *us, *theta;
  Matrix *brels_mat, *ibabs_mat, *bpos_mat, *weights_mat, *points_mat;
  Triangle *triangles_mat;
} HandInput;

/* Store the converted results to matrices for Enzyme */
struct MatrixConverted {
  Matrix *base_relatives, *inverse_base_absolutes, *base_positions, *weights,
      *points;
  Triangle *triangles;
};
