#pragma once
#include "hand_types.h"
#include "lagrad_utils.h"
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double *deadbeef = (double *)0xdeadbeef;

typedef struct HandComplicatedGrad {
  F64Descriptor1D dtheta;
  F64Descriptor2D dus;
} HandComplicatedGrad;

void transpose_in_place(double *matrix, size_t n) {
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      double tmp = matrix[i * n + j];
      matrix[i * n + j] = matrix[j * n + i];
      matrix[j * n + i] = tmp;
    }
  }
}

Matrix *ptr_to_matrices(double *data, size_t num_matrices, size_t m, size_t n) {
  Matrix *matrices = (Matrix *)malloc(num_matrices * sizeof(Matrix));
  int stride = m * n;
  for (size_t mat = 0; mat < num_matrices; mat++) {
    matrices[mat].nrows = m;
    matrices[mat].ncols = n;
    matrices[mat].data = &data[mat * stride];
  }
  return matrices;
}

Matrix *ptr_to_matrix(double *data, size_t m, size_t n) {
  Matrix *matrix = malloc(sizeof(Matrix));
  matrix->nrows = m;
  matrix->ncols = n;
  matrix->data = data;
  return matrix;
}

Triangle *ptr_to_triangles(int *data, size_t num_triangles) {
  Triangle *triangles = (Triangle *)malloc(num_triangles * sizeof(Triangle));
  for (size_t tri = 0; tri < num_triangles; tri++) {
    triangles[tri].verts[0] = data[tri * 3 + 0];
    triangles[tri].verts[1] = data[tri * 3 + 1];
    triangles[tri].verts[2] = data[tri * 3 + 2];
  }
  return triangles;
}

void free_matrix_array(Matrix *matrices, size_t num_matrices) {
  // for (size_t i = 0; i < num_matrices; i++) {
  //   free(matrices[i].data);
  // }
  free(matrices);
}

HandModel read_hand_model(const char *model_path, bool transpose) {
  const char DELIMITER = ':';
  char filename[120];
  char *currentline = (char *)malloc(100 * sizeof(char));
  FILE *fp;

  strcpy(filename, model_path);
  strcat(filename, "/bones.txt");
  fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Failed to open file \"%s\"\n", filename);
    exit(EXIT_FAILURE);
  }

  // Could be annoying if this ever changes, but it's not worth the hassle of
  // making this dynamic.
  const int n_bones = 22;
  const char **bone_names = (const char **)malloc(n_bones * sizeof(char *));
  int *parents = (int *)malloc(n_bones * sizeof(int));
  double *base_relatives = (double *)malloc(n_bones * 16 * sizeof(double));
  double *inverse_base_absolutes =
      (double *)malloc(n_bones * 16 * sizeof(double));
  size_t n;
  // These are also column major
  for (size_t i = 0; i < n_bones; i++) {
    getdelim(&currentline, &n, DELIMITER, fp);
    size_t str_size = strlen(currentline);
    char *name = (char *)malloc(str_size * sizeof(char));
    strncpy(name, currentline, str_size - 1);
    bone_names[i] = name;

    getdelim(&currentline, &n, DELIMITER, fp);
    parents[i] = strtol(currentline, NULL, 10);

    for (size_t j = 0; j < 16; j++) {
      getdelim(&currentline, &n, DELIMITER, fp);
      base_relatives[i * 16 + j] = strtod(currentline, NULL);
    }
    if (transpose) {
      transpose_in_place(&base_relatives[i * 16], 4);
    }

    for (size_t j = 0; j < 15; j++) {
      getdelim(&currentline, &n, DELIMITER, fp);
      inverse_base_absolutes[i * 16 + j] = strtod(currentline, NULL);
    }
    getdelim(&currentline, &n, '\n', fp);
    inverse_base_absolutes[i * 16 + 15] = strtod(currentline, NULL);
    if (transpose) {
      transpose_in_place(&inverse_base_absolutes[i * 16], 4);
    }
  }

  fclose(fp);
  filename[0] = '\0';
  strcpy(filename, model_path);
  strcat(filename, "/vertices.txt");
  fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Failed to open file \"%s\"\n", filename);
    exit(EXIT_FAILURE);
  }

  int n_vertices = 0;
  while (getline(&currentline, &n, fp) != -1) {
    if (currentline[0] != '\0')
      n_vertices++;
  }
  if (errno != 0) {
    fprintf(stderr, "Something went wrong reading \"%s\" (%d)", filename,
            errno);
    exit(EXIT_FAILURE);
  }
  fclose(fp);

  double *base_positions = (double *)malloc(4 * n_vertices * sizeof(double));
  double *weights = (double *)malloc(n_bones * n_vertices * sizeof(double));
  for (size_t i = 0; i < n_bones * n_vertices; i++) {
    weights[i] = 0.0;
  }

  fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Failed to open file \"%s\"\n", filename);
    exit(EXIT_FAILURE);
  }

  for (size_t i_vert = 0; i_vert < n_vertices; i_vert++) {
    for (size_t j = 0; j < 3; j++) {
      getdelim(&currentline, &n, DELIMITER, fp);
      if (transpose) {
        base_positions[j + i_vert * 4] = strtod(currentline, NULL);
      } else {
        base_positions[j * n_vertices + i_vert] = strtod(currentline, NULL);
      }
    }
    if (transpose) {
      base_positions[3 + i_vert * 4] = 1.0;
    } else {
      base_positions[3 * n_vertices + i_vert] = 1.0;
    }
    for (size_t j = 0; j < 3 + 2; j++) {
      getdelim(&currentline, &n, DELIMITER, fp); // Skip
    }
    getdelim(&currentline, &n, DELIMITER, fp);
    int n0 = strtol(currentline, NULL, 10);
    for (size_t j = 0; j < n0; j++) {
      getdelim(&currentline, &n, DELIMITER, fp);
      int i_bone = strtol(currentline, NULL, 10);
      if (j == n0 - 1)
        getdelim(&currentline, &n, '\n', fp);
      else
        getdelim(&currentline, &n, DELIMITER, fp);
      weights[i_bone + i_vert * n_bones] = strtod(currentline, NULL);
    }
  }

  fclose(fp);
  filename[0] = '\0';
  strcpy(filename, model_path);
  strcat(filename, "/triangles.txt");

  fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Failed to open file \"%s\"\n", filename);
    exit(EXIT_FAILURE);
  }

  // Also not worth making this dynamic.
  const size_t n_triangles = 1084;
  int *triangles = (int *)malloc(n_triangles * 3 * sizeof(int));
  for (size_t tri = 0; tri < n_triangles; tri++) {
    getdelim(&currentline, &n, DELIMITER, fp);
    if (currentline[0] == '\0')
      continue;
    triangles[tri * 3 + 0] = strtol(currentline, NULL, 10);
    getdelim(&currentline, &n, DELIMITER, fp);
    triangles[tri * 3 + 1] = strtol(currentline, NULL, 10);
    getdelim(&currentline, &n, '\n', fp);
    triangles[tri * 3 + 2] = strtol(currentline, NULL, 10);
  }
  fclose(fp);

  HandModel model = {.n_bones = n_bones,
                     .n_vertices = n_vertices,
                     .n_triangles = n_triangles,
                     .bone_names = bone_names,
                     .parents = parents,
                     .base_relatives = base_relatives,
                     .inverse_base_absolutes = inverse_base_absolutes,
                     .base_positions = base_positions,
                     .weights = weights,
                     .triangles = triangles,
                     .is_mirrored = false};
  free(currentline);
  return model;
}

void free_hand_model(HandModel *model) {
  // TODO: Freeing this is causing segvs for some reason. It's leaking memory
  // but it's not the biggest deal.

  // for (size_t i = 0; i < model->n_bones; i++)
  // {
  //   free((char *)model->bone_names[i]);
  // }
  free(model->bone_names);
  free(model->parents);
  free(model->base_relatives);
  free(model->inverse_base_absolutes);
  free(model->base_positions);
  free(model->weights);
  free(model->triangles);
}

HandInput read_hand_data(const char *model_path, const char *data_file,
                         bool complicated, bool transpose) {
  HandModel model = read_hand_model(model_path, transpose);
  FILE *fp = fopen(data_file, "r");
  if (!fp) {
    fprintf(stderr, "Failed to open file \"%s\"\n", data_file);
    exit(EXIT_FAILURE);
  }

  int n_pts, n_theta;
  fscanf(fp, "%d %d", &n_pts, &n_theta);

  int *correspondences = (int *)malloc(n_pts * sizeof(int));
  double *points = (double *)malloc(3 * n_pts * sizeof(double));

  // This is read in column major order.
  for (size_t i = 0; i < n_pts; i++) {
    fscanf(fp, "%d", &correspondences[i]);
    for (size_t j = 0; j < 3; j++) {
      fscanf(fp, "%lf", &points[j + i * 3]);
    }
  }

  double *us = NULL;
  if (complicated) {
    us = (double *)malloc(2 * n_pts * sizeof(double));
    for (size_t i = 0; i < 2 * n_pts; i++) {
      fscanf(fp, "%lf", &us[i]);
    }
  }

  double *theta = (double *)malloc(n_theta * sizeof(double));
  for (size_t i = 0; i < n_theta; i++) {
    fscanf(fp, "%lf", &theta[i]);
  }

  Matrix *brels_mat =
      ptr_to_matrices(model.base_relatives, model.n_bones, 4, 4);
  Matrix *ibabs_mat =
      ptr_to_matrices(model.inverse_base_absolutes, model.n_bones, 4, 4);
  Matrix *bpos_mat = ptr_to_matrix(model.base_positions, 4, model.n_vertices);
  Matrix *weights_mat =
      ptr_to_matrix(model.weights, model.n_bones, model.n_vertices);
  Matrix *points_mat = ptr_to_matrix(points, 3, n_pts);
  Triangle *triangles_mat =
      ptr_to_triangles(model.triangles, model.n_triangles);

  HandInput hand_input = {.model = model,
                          .correspondences = correspondences,
                          .n_theta = n_theta,
                          .n_pts = n_pts,
                          .points = points,
                          .us = us,
                          .theta = theta,
                          .brels_mat = brels_mat,
                          .ibabs_mat = ibabs_mat,
                          .bpos_mat = bpos_mat,
                          .weights_mat = weights_mat,
                          .points_mat = points_mat,
                          .triangles_mat = triangles_mat};
  fclose(fp);
  return hand_input;
}

void free_hand_input(HandInput *input) {
  free_hand_model(&input->model);
  free(input->correspondences);
  free(input->points);
  free(input->theta);
  if (input->us != NULL) {
    free(input->us);
  }
}

void parse_hand_results(const char *ffile, double *J, size_t m, size_t n) {
  FILE *fp = fopen(ffile, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file \"%s\"\n", ffile);
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      fscanf(fp, "%lf", &J[i * n + j]);
    }
  }

  fclose(fp);
}

void verify_hand_results(double *ref_J, double *J, size_t m, size_t n,
                         const char *application) {
  double err = 0.0;
  for (size_t i = 0; i < m * n; i++) {
    err += fabs(ref_J[i] - J[i]);
  }
  if (err > 1e-6) {
    printf("(%s) Hand Jacobian error: %f\n", application, err);
  }
}

F64Descriptor1D lagrad_hand_objective(
    /*theta=*/double *, double *, int64_t, int64_t, int64_t,
    /*parents=*/int32_t *, int32_t *, int64_t, int64_t, int64_t,
    /*base_relatives=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    /*inverse_base_absolutes=*/double *, double *, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t,
    /*base_positions=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*weights=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*correspondence=*/int32_t *, int32_t *, int64_t, int64_t, int64_t,
    /*points=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*g=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t
    //);
    ,
    /*indices=*/int64_t *, int64_t *, int64_t, int64_t, int64_t);

F64Descriptor1D lagrad_hand_simple(HandInput *input, double *derr, int row,
                                   int col) {
  int64_t indices[2] = {row, col};
  return lagrad_hand_objective(
      /*theta=*/deadbeef, input->theta, 0, input->n_theta, 1,
      /*parents=*/(int32_t *)deadbeef, input->model.parents, 0,
      input->model.n_bones, 1,
      /*base_relatives=*/deadbeef, input->model.base_relatives, 0,
      input->model.n_bones, 4, 4, 16, 4, 1,
      /*inverse_base_absolutes=*/deadbeef, input->model.inverse_base_absolutes,
      0, input->model.n_bones, 4, 4, 16, 4, 1,
      /*base_positions=*/deadbeef, input->model.base_positions, 0,
      input->model.n_vertices, 4, 4, 1,
      /*weights=*/deadbeef, input->model.weights, 0, input->model.n_vertices,
      input->model.n_bones, input->model.n_bones, 1,
      /*correspondences=*/(int32_t *)deadbeef, input->correspondences, 0,
      input->n_pts, 1,
      /*points=*/deadbeef, input->points, 0, input->n_pts, 3, 3, 1,
      /*g=*/deadbeef, derr, 0, input->n_pts, 3, 3,
      1
      // );
      ,
      /*indices=*/(int64_t *)deadbeef, indices, 0, 2, 1);
}

F64Descriptor1D sparse_grad_mlir_hand_objective(
    /*theta=*/double *, double *, int64_t, int64_t, int64_t,
    /*parents=*/int32_t *, int32_t *, int64_t, int64_t, int64_t,
    /*base_relatives=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    /*inverse_base_absolutes=*/double *, double *, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t,
    /*base_positions=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*weights=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*correspondence=*/int32_t *, int32_t *, int64_t, int64_t, int64_t,
    /*points=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*g=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);

F64Descriptor1D sparse_lagrad_hand_simple(HandInput *input, double *derr) {
  return sparse_grad_mlir_hand_objective(
      /*theta=*/deadbeef, input->theta, 0, input->n_theta, 1,
      /*parents=*/(int32_t *)deadbeef, input->model.parents, 0,
      input->model.n_bones, 1,
      /*base_relatives=*/deadbeef, input->model.base_relatives, 0,
      input->model.n_bones, 4, 4, 16, 4, 1,
      /*inverse_base_absolutes=*/deadbeef, input->model.inverse_base_absolutes,
      0, input->model.n_bones, 4, 4, 16, 4, 1,
      /*base_positions=*/deadbeef, input->model.base_positions, 0,
      input->model.n_vertices, 4, 4, 1,
      /*weights=*/deadbeef, input->model.weights, 0, input->model.n_vertices,
      input->model.n_bones, input->model.n_bones, 1,
      /*correspondences=*/(int32_t *)deadbeef, input->correspondences, 0,
      input->n_pts, 1,
      /*points=*/deadbeef, input->points, 0, input->n_pts, 3, 3, 1,
      /*g=*/deadbeef, derr, 0, input->n_pts, 3, 3, 1);
}

HandComplicatedGrad lagrad_hand_objective_complicated(
    /*theta=*/double *, double *, int64_t, int64_t, int64_t,
    /*us=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*parents=*/int32_t *, int32_t *, int64_t, int64_t, int64_t,
    /*base_relatives=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    /*inverse_base_absolutes=*/double *, double *, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t,
    /*base_positions=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*weights=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*triangles=*/int32_t *, int32_t *, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*correspondence=*/int32_t *, int32_t *, int64_t, int64_t, int64_t,
    /*points=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*g=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);

HandComplicatedGrad lagrad_hand_complicated(HandInput *input, double *derr) {
  return lagrad_hand_objective_complicated(
      /*theta=*/deadbeef, input->theta, 0, input->n_theta, 1,
      /*us=*/deadbeef, input->us, 0, input->n_pts, 2, 2, 1,
      /*parents=*/(int32_t *)deadbeef, input->model.parents, 0,
      input->model.n_bones, 1,
      /*base_relatives=*/deadbeef, input->model.base_relatives, 0,
      input->model.n_bones, 4, 4, 16, 4, 1,
      /*inverse_base_absolutes=*/deadbeef, input->model.inverse_base_absolutes,
      0, input->model.n_bones, 4, 4, 16, 4, 1,
      /*base_positions=*/deadbeef, input->model.base_positions, 0,
      input->model.n_vertices, 4, 4, 1,
      /*weights=*/deadbeef, input->model.weights, 0, input->model.n_vertices,
      input->model.n_bones, input->model.n_bones, 1,
      /*triangles=*/(int32_t *)deadbeef, input->model.triangles, 0,
      input->model.n_triangles, 3, 3, 1,
      /*correspondences=*/(int32_t *)deadbeef, input->correspondences, 0,
      input->n_pts, 1,
      /*points=*/deadbeef, input->points, 0, input->n_pts, 3, 3, 1,
      /*g=*/deadbeef, derr, 0, input->n_pts, 3, 3, 1);
}

F64Descriptor1D enzyme_hand_objective(
    /*theta=*/double *, double *, int64_t, int64_t, int64_t,
    /*parents=*/int32_t *, int32_t *, int64_t, int64_t, int64_t,
    /*base_relatives=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    /*inverse_base_absolutes=*/double *, double *, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t,
    /*base_positions=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*weights=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*correspondence=*/int32_t *, int32_t *, int64_t, int64_t, int64_t,
    /*points=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*derr=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);

F64Descriptor1D enzyme_mlir_hand_simple(HandInput *input, double *derr, int row,
                                        int col) {
  return enzyme_hand_objective(
      /*theta=*/deadbeef, input->theta, 0, input->n_theta, 1,
      /*parents=*/(int32_t *)deadbeef, input->model.parents, 0,
      input->model.n_bones, 1,
      /*base_relatives=*/deadbeef, input->model.base_relatives, 0,
      input->model.n_bones, 4, 4, 16, 4, 1,
      /*inverse_base_absolutes=*/deadbeef, input->model.inverse_base_absolutes,
      0, input->model.n_bones, 4, 4, 16, 4, 1,
      /*base_positions=*/deadbeef, input->model.base_positions, 0,
      input->model.n_vertices, 4, 4, 1,
      /*weights=*/deadbeef, input->model.weights, 0, input->model.n_vertices,
      input->model.n_bones, input->model.n_bones, 1,
      /*correspondences=*/(int32_t *)deadbeef, input->correspondences, 0,
      input->n_pts, 1,
      /*points=*/deadbeef, input->points, 0, input->n_pts, 3, 3, 1,
      /*g=*/deadbeef, derr, 0, input->n_pts, 3, 3, 1);
}

HandComplicatedGrad enzyme_hand_objective_complicated(
    /*theta=*/double *, double *, int64_t, int64_t, int64_t,
    /*us=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*parents=*/int32_t *, int32_t *, int64_t, int64_t, int64_t,
    /*base_relatives=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    /*inverse_base_absolutes=*/double *, double *, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t,
    /*base_positions=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*weights=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*triangles=*/int32_t *, int32_t *, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*correspondence=*/int32_t *, int32_t *, int64_t, int64_t, int64_t,
    /*points=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t,
    /*g=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);

HandComplicatedGrad enzyme_mlir_hand_complicated(HandInput *input,
                                                 double *derr) {
  return enzyme_hand_objective_complicated(
      /*theta=*/deadbeef, input->theta, 0, input->n_theta, 1,
      /*us=*/deadbeef, input->us, 0, input->n_pts, 2, 2, 1,
      /*parents=*/(int32_t *)deadbeef, input->model.parents, 0,
      input->model.n_bones, 1,
      /*base_relatives=*/deadbeef, input->model.base_relatives, 0,
      input->model.n_bones, 4, 4, 16, 4, 1,
      /*inverse_base_absolutes=*/deadbeef, input->model.inverse_base_absolutes,
      0, input->model.n_bones, 4, 4, 16, 4, 1,
      /*base_positions=*/deadbeef, input->model.base_positions, 0,
      input->model.n_vertices, 4, 4, 1,
      /*weights=*/deadbeef, input->model.weights, 0, input->model.n_vertices,
      input->model.n_bones, input->model.n_bones, 1,
      /*triangles=*/(int32_t *)deadbeef, input->model.triangles, 0,
      input->model.n_triangles, 3, 3, 1,
      /*correspondences=*/(int32_t *)deadbeef, input->correspondences, 0,
      input->n_pts, 1,
      /*points=*/deadbeef, input->points, 0, input->n_pts, 3, 3, 1,
      /*g=*/deadbeef, derr, 0, input->n_pts, 3, 3, 1);
}

void dhand_objective(double const *theta, double *dtheta, int bone_count,
                     const char **bone_names, const int *parents,
                     Matrix *base_relatives, Matrix *inverse_base_absolutes,
                     Matrix *base_positions, Matrix *weights,
                     const Triangle *triangles, int is_mirrored,
                     int corresp_count, const int *correspondences,
                     Matrix *points, double *err, double *derr);

F64Descriptor1D enzyme_c_hand_simple(HandInput *input, double *derr, int row,
                                     int col) {
  F64Descriptor1D dtheta = {.allocated = NULL,
                            .aligned = calloc(input->n_theta, sizeof(double)),
                            .offset = 0,
                            .size = input->n_theta,
                            .stride = 1};
  double *err = malloc(input->n_pts * 3 * sizeof(double));
  dhand_objective(input->theta, dtheta.aligned, input->model.n_bones,
                  input->model.bone_names, input->model.parents,
                  input->brels_mat, input->ibabs_mat, input->bpos_mat,
                  input->weights_mat, input->triangles_mat, 0, input->n_pts,
                  input->correspondences, input->points_mat, err, derr);
  free(err);
  return dtheta;
}
