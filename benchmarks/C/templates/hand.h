#pragma once
#include "mlir_c_abi.h"
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HAND_MODEL_PATH "benchmarks/data/hand_model"
#define HAND_DATA_FILE "benchmarks/data/hand1_t26_c100.txt"

typedef struct {
  int n_bones;
  char **bone_names;
  int *parents;
  double *base_relatives, *inverse_base_absolutes, *base_positions, *weights;
  int *triangles;
  bool is_mirrored;
} HandModel;

typedef struct {
  HandModel model;
  int *correspondences;
  int n_theta;
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
  char **bone_names = (char **)malloc(n_bones * sizeof(char *));
  int *parents = (int *)malloc(n_bones * sizeof(int));
  double *base_relatives = (double *)malloc(n_bones * 16 * sizeof(double));
  double *inverse_base_absolutes =
      (double *)malloc(n_bones * 16 * sizeof(double));
  size_t n;
  for (size_t i = 0; i < n_bones; i++) {
    getdelim(&currentline, &n, DELIMITER, fp);
    size_t str_size = strlen(currentline);
    char *name = malloc(str_size * sizeof(char));
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
      base_positions[j * n_vertices + i_vert] = strtod(currentline, NULL);
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
      weights[i_bone * n_vertices + i_vert] = strtod(currentline, NULL);
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
    free(model->bone_names[i]);
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

  for (size_t i = 0; i < n_pts; i++) {
    fscanf(fp, "%d", &correspondences[i]);
    for (size_t j = 0; j < 3; j++) {
      fscanf(fp, "%lf", &points[j * n_pts + i]);
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
                          .points = points,
                          .us = us,
                          .theta = theta};
  fclose(fp);
  return hand_input;
}

void free_hand_input(HandInput *input) {
  free(input->correspondences);
  free(input->points);
  free(input->theta);
  if (input->us != NULL) {
    free(input->us);
  }
}

// TODO: Currently incomplete.
void enzyme_calculate_jacobian_simple(HandInput *input) {
  double *theta_d = (double *)malloc(input->n_theta * sizeof(double));
  for (size_t i = 0; i < input->n_theta; i++) {
    theta_d[i] = 0;
  }

  for (size_t i = 0; i < input->n_theta; i++) {
    if (i > 0) {
      theta_d[i - 1] = 0.0;
    }
    theta_d[i] = 1.0;

    free(theta_d);
  }
}
