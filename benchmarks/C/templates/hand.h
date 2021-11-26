#pragma once
#include "mlir_c_abi.h"
#include "shared_types.h"
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HAND_MODEL_PATH "benchmarks/data/hand_model"
#define HAND_DATA_FILE "benchmarks/data/test.txt"

typedef struct {
  int n_bones, n_vertices;
  const char **bone_names;
  int *parents;
  double *base_relatives, *inverse_base_absolutes, *base_positions, *weights;
  int *triangles;
  bool is_mirrored;
} HandModel;

typedef struct {
  HandModel model;
  int *correspondences;
  int n_theta, n_pts;
  double *points, *us, *theta;
} HandInput;

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
    matrices[mat].data = (double *)malloc(m * n * sizeof(double));
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        matrices[mat].data[i * n + j] = data[mat * stride + i * n + j];
      }
    }
  }
  return matrices;
}

Matrix ptr_to_matrix(double *data, size_t m, size_t n) {
  double *mdata = (double *)malloc(m * n * sizeof(double));
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      mdata[i * n + j] = data[i * n + j];
    }
  }

  Matrix matrix = {.nrows = m, .ncols = n, .data = mdata};
  return matrix;
}

void free_matrix_array(Matrix *matrices, size_t num_matrices) {
  for (size_t i = 0; i < num_matrices; i++) {
    free(matrices[i].data);
  }
  free(matrices);
}

HandModel read_hand_model() {
  const char DELIMITER = ':';
  char filename[80];
  char *currentline = (char *)malloc(100 * sizeof(char));
  FILE *fp;

  strcpy(filename, HAND_MODEL_PATH);
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
    transpose_in_place(&base_relatives[i * 16], 4);

    for (size_t j = 0; j < 15; j++) {
      getdelim(&currentline, &n, DELIMITER, fp);
      inverse_base_absolutes[i * 16 + j] = strtod(currentline, NULL);
    }
    getdelim(&currentline, &n, '\n', fp);
    inverse_base_absolutes[i * 16 + 15] = strtod(currentline, NULL);
    transpose_in_place(&inverse_base_absolutes[i * 16], 4);
  }

  fclose(fp);
  filename[0] = '\0';
  strcpy(filename, HAND_MODEL_PATH);
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
      base_positions[j + i_vert * 4] = strtod(currentline, NULL);
    }
    base_positions[3 + i_vert * 4] = 1.0;
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
  strcpy(filename, HAND_MODEL_PATH);
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
  for (size_t i = 0; i < model->n_bones; i++) {
    free((char *)model->bone_names[i]);
  }
  free(model->bone_names);
  free(model->parents);
  free(model->base_relatives);
  free(model->inverse_base_absolutes);
  free(model->base_positions);
  free(model->weights);
  free(model->triangles);
}

HandInput read_hand_data(bool complicated) {
  HandModel model = read_hand_model();
  FILE *fp = fopen(HAND_DATA_FILE, "r");
  if (!fp) {
    fprintf(stderr, "Failed to open file \"%s\"\n", HAND_DATA_FILE);
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

  HandInput hand_input = {.model = model,
                          .correspondences = correspondences,
                          .n_theta = n_theta,
                          .n_pts = n_pts,
                          .points = points,
                          .us = us,
                          .theta = theta};
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

/**
 * @brief Serialize a 2d Jacobian matrix in row-major order.
 *
 * @param ffile the name of the file to write to
 * @param J the Jacobian matrix
 * @param m the number of rows
 * @param n the number of columns
 */
void serialize_hand_results(const char *ffile, double *J, size_t m, size_t n) {
  FILE *fp = fopen(ffile, "w");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file \"%s\"\n", ffile);
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      fprintf(fp, "%.10e", J[i * n + j]);
      if (j != n - 1) {
        fprintf(fp, " ");
      }
    }
    fprintf(fp, "\n");
  }
  fprintf(fp, "\n");

  fclose(fp);
}
// TODO: Currently incomplete.
// void enzyme_calculate_jacobian_simple(HandInput *input) {
//   double *theta_d = (double *)malloc(input->n_theta * sizeof(double));
//   for (size_t i = 0; i < input->n_theta; i++) {
//     theta_d[i] = 0;
//   }

//   for (size_t i = 0; i < input->n_theta; i++) {
//     if (i > 0) {
//       theta_d[i - 1] = 0.0;
//     }
//     theta_d[i] = 1.0;

//     free(theta_d);
//   }
// }
