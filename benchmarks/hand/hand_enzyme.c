#include "hand_types.h"
#include "lagrad_utils.h"
#include <math.h>
#include <stdlib.h>

/*========================================================================*/
/*                            MATRIX METHODS                              */
/*========================================================================*/
// returns a new matrix of size nrows * ncols
Matrix *get_new_matrix(int nrows, int ncols) {
  Matrix *mat = (Matrix *)malloc(sizeof(Matrix));

  mat->nrows = nrows;
  mat->ncols = ncols;
  mat->data = (double *)malloc(nrows * ncols * sizeof(double));

  return mat;
}

// return new empty matrix
Matrix *get_new_empty_matrix() {
  Matrix *mat = (Matrix *)malloc(sizeof(Matrix));

  mat->nrows = 0;
  mat->ncols = 0;
  mat->data = NULL;

  return mat;
}

// disposes data matrix holds
void delete_matrix(Matrix *mat) {
  if (mat->data != NULL) {
    free(mat->data);
  }

  free(mat);
}

// creates array of matricies
Matrix *get_matrix_array(int count) {
  Matrix *result = (Matrix *)malloc(count * sizeof(Matrix));

  int i;
  for (i = 0; i < count; i++) {
    result[i].data = NULL;
    result[i].nrows = result[i].ncols = 0;
  }

  return result;
}

// disposes array of matricies
void delete_light_matrix_array(Matrix *matricies, int count) {
  int i;
  for (i = 0; i < count; i++) {
    if (matricies[i].data != NULL) {
      free(matricies[i].data);
    }
  }

  free(matricies);
}

// sets a new size of a matrix
void resize(Matrix *mat, int nrows, int ncols) {
  if (mat->nrows * mat->ncols != nrows * ncols) {
    if (mat->data != NULL) {
      free(mat->data);
    }

    if (nrows * ncols > 0) {
      mat->data = (double *)malloc(ncols * nrows * sizeof(double));
    } else {
      mat->data = NULL;
    }
  }

  mat->ncols = ncols;
  mat->nrows = nrows;
}

// multiplies matricies
void mat_mult(const Matrix *__restrict lhs, const Matrix *__restrict rhs,
              Matrix *__restrict out) {
  int i, j, k;
  resize(out, lhs->nrows, rhs->ncols);
  for (i = 0; i < lhs->nrows; i++) {
    for (k = 0; k < rhs->ncols; k++) {
      out->data[i + k * out->nrows] =
          lhs->data[i + 0 * lhs->nrows] * rhs->data[0 + k * rhs->nrows];
      for (j = 1; j < lhs->ncols; j++) {
        out->data[i + k * out->nrows] +=
            lhs->data[i + j * lhs->nrows] * rhs->data[j + k * rhs->nrows];
      }
    }
  }
}

// set matrix identity
void set_identity(Matrix *mat) {
  int i_col, i_row;
  for (i_col = 0; i_col < mat->ncols; i_col++) {
    for (i_row = 0; i_row < mat->nrows; i_row++) {
      if (i_col == i_row) {
        mat->data[i_row + i_col * mat->nrows] = 1.0;
      } else {
        mat->data[i_row + i_col * mat->nrows] = 0.0;
      }
    }
  }
}

// fills matrix with the given value
void fill(Matrix *mat, double val) {
  int i;
  for (i = 0; i < mat->ncols * mat->nrows; i++) {
    mat->data[i] = val;
  }
}

// set a block of the matrix with another matrix
void set_block(Matrix *mat, int row_off, int col_off, const Matrix *block) {
  int i_col, i_row;
  for (i_col = 0; i_col < block->ncols; i_col++) {
    for (i_row = 0; i_row < block->nrows; i_row++) {
      mat->data[i_row + row_off + (i_col + col_off) * mat->nrows] =
          block->data[i_row + i_col * block->nrows];
    }
  }
}

// copies one matrix to another
void copy(Matrix *dst, const Matrix *src) {
  if (dst->data != NULL) {
    free(dst->data);
  }

  dst->ncols = src->ncols;
  dst->nrows = src->nrows;
  dst->data = (double *)malloc(dst->ncols * dst->nrows * sizeof(double));

  int i;
  for (i = 0; i < dst->ncols * dst->nrows; i++) {
    dst->data[i] = src->data[i];
  }
}

/*========================================================================*/
/*                                   UTILS                                */
/*========================================================================*/
// sum of component squares
double square_sum(int n, double const *x) {
  int i;
  double res = x[0] * x[0];
  for (i = 1; i < n; i++) {
    res = res + x[i] * x[i];
  }

  return res;
}

/*========================================================================*/
/*                              MAIN LOGIC                                */
/*========================================================================*/
void angle_axis_to_rotation_matrix(double const *angle_axis, Matrix *R) {
  double norm = sqrt(square_sum(3, angle_axis));
  if (norm < 0.0001) {
    set_identity(R);
    return;
  }

  double x = angle_axis[0] / norm;
  double y = angle_axis[1] / norm;
  double z = angle_axis[2] / norm;

  double s = sin(norm);
  double c = cos(norm);

  R->data[0 + 0 * R->nrows] = x * x + (1 - x * x) * c; // first row
  R->data[0 + 1 * R->nrows] = x * y * (1 - c) - z * s;
  R->data[0 + 2 * R->nrows] = x * z * (1 - c) + y * s;

  R->data[1 + 0 * R->nrows] = x * y * (1 - c) + z * s; // second row
  R->data[1 + 1 * R->nrows] = y * y + (1 - y * y) * c;
  R->data[1 + 2 * R->nrows] = y * z * (1 - c) - x * s;

  R->data[2 + 0 * R->nrows] = x * z * (1 - c) - y * s; // third row
  R->data[2 + 1 * R->nrows] = z * y * (1 - c) + x * s;
  R->data[2 + 2 * R->nrows] = z * z + (1 - z * z) * c;
}

void apply_global_transform(const Matrix *pose_params, Matrix *positions) {
  int i, j;
  Matrix *R = get_new_matrix(3, 3);
  angle_axis_to_rotation_matrix(pose_params->data, R);
  // nrows is 3, ncols is 544

  // This just multiplies each entry by 1.
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      R->data[j + i * R->nrows] *=
          pose_params->data[i + 1 * pose_params->nrows];
    }
  }
  // R is unchanged

  Matrix *tmp = get_new_empty_matrix();
  mat_mult(R, positions, tmp);
  for (j = 0; j < positions->ncols; j++) {
    for (i = 0; i < positions->nrows; i++) {
      positions->data[i + j * positions->nrows] =
          tmp->data[i + j * tmp->nrows] +
          pose_params->data[i + 2 * pose_params->nrows];
    }
  }

  delete_matrix(R);
  delete_matrix(tmp);
}

void relatives_to_absolutes(int count, const Matrix *relatives,
                            const int *parents, Matrix *absolutes) {
  int i;
  for (i = 0; i < count; i++) {
    if (parents[i] == -1) {
      copy(&absolutes[i], &relatives[i]);
    } else {
      mat_mult(&absolutes[parents[i]], &relatives[i], &absolutes[i]);
    }
  }
}

void euler_angles_to_rotation_matrix(double const *__restrict xzy,
                                     Matrix *__restrict R) {
  double tx = xzy[0];
  double ty = xzy[2];
  double tz = xzy[1];

  Matrix *Rx = get_new_matrix(3, 3);
  Matrix *Ry = get_new_matrix(3, 3);
  Matrix *Rz = get_new_matrix(3, 3);

  set_identity(Rx);
  Rx->data[1 + 1 * Rx->nrows] = cos(tx);
  Rx->data[2 + 1 * Rx->nrows] = sin(tx);
  Rx->data[1 + 2 * Rx->nrows] = -Rx->data[2 + 1 * Rx->nrows];
  Rx->data[2 + 2 * Rx->nrows] = Rx->data[1 + 1 * Rx->nrows];

  set_identity(Ry);
  Ry->data[0 + 0 * Ry->nrows] = cos(ty);
  Ry->data[0 + 2 * Ry->nrows] = sin(ty);
  Ry->data[2 + 0 * Ry->nrows] = -Ry->data[0 + 2 * Ry->nrows];
  Ry->data[2 + 2 * Ry->nrows] = Ry->data[0 + 0 * Ry->nrows];

  set_identity(Rz);
  Rz->data[0 + 0 * Rz->nrows] = cos(tz);
  Rz->data[1 + 0 * Rz->nrows] = sin(tz);
  Rz->data[0 + 1 * Rz->nrows] = -Rz->data[1 + 0 * Rz->nrows];
  Rz->data[1 + 1 * Rz->nrows] = Rz->data[0 + 0 * Rz->nrows];

  Matrix *tmp = get_new_empty_matrix();
  mat_mult(Rz, Ry, tmp);
  mat_mult(tmp, Rx, R);

  delete_matrix(Rx);
  delete_matrix(Ry);
  delete_matrix(Rz);
  delete_matrix(tmp);
}

void get_posed_relatives(int bone_count,
                         const Matrix *__restrict base_relatives,
                         const Matrix *__restrict pose_params,
                         Matrix *__restrict relatives) {
  int i;
  int offset = 3;
  Matrix *tr = get_new_matrix(4, 4);
  Matrix *R = get_new_matrix(3, 3);

  for (i = 0; i < bone_count; i++) {
    set_identity(tr);

    euler_angles_to_rotation_matrix(
        pose_params->data + (i + offset) * pose_params->nrows, R);
    set_block(tr, 0, 0, R);

    mat_mult(&base_relatives[i], tr, &relatives[i]);
  }

  delete_matrix(tr);
  delete_matrix(R);
}

/* The 4x4 matrices here are transposed (column major) relative to ADBench */
static inline void get_skinned_vertex_positions(
    int bone_count, const Matrix *__restrict base_relatives, const int *parents,
    const Matrix *__restrict inverse_base_absolutes,
    const Matrix *__restrict base_positions, const Matrix *__restrict weights,
    int is_mirrored, const Matrix *__restrict pose_params,
    Matrix *__restrict positions, int apply_global) {
  int i;

  Matrix *relatives = get_matrix_array(bone_count);
  Matrix *absolutes = get_matrix_array(bone_count);
  Matrix *transforms = get_matrix_array(bone_count);

  get_posed_relatives(bone_count, base_relatives, pose_params, relatives);
  relatives_to_absolutes(bone_count, relatives, parents, absolutes);
  // Get bone transforms->
  for (i = 0; i < bone_count; i++) {
    mat_mult(&absolutes[i], &inverse_base_absolutes[i], &transforms[i]);
  }

  // Transform vertices by necessary transforms-> + apply skinning
  /* ncols is 544 */
  resize(positions, 3, base_positions->ncols);
  fill(positions, 0.0);
  Matrix *curr_positions = get_new_matrix(4, base_positions->ncols);

  int i_bone, i_vert;
  for (i_bone = 0; i_bone < bone_count; i_bone++) {
    mat_mult(&transforms[i_bone], base_positions, curr_positions);
    /* positions->ncols is also 544 */
    for (i_vert = 0; i_vert < positions->ncols; i_vert++) {
      for (i = 0; i < 3; i++) {
        positions->data[i + i_vert * positions->nrows] +=
            curr_positions->data[i + i_vert * curr_positions->nrows] *
            weights->data[i_bone + i_vert * weights->nrows];
      }
    }
  }

  /* Always false */
  if (is_mirrored) {
    for (i = 0; i < positions->ncols; i++) {
      positions->data[0 + i * positions->nrows] *= -1;
    }
  }

  /* Always true */
  if (apply_global) {
    apply_global_transform(pose_params, positions);
  }

  delete_matrix(curr_positions);
  delete_light_matrix_array(relatives, bone_count);
  delete_light_matrix_array(absolutes, bone_count);
  delete_light_matrix_array(transforms, bone_count);
}

//% !!!!!!! fixed order pose_params !!!!!
//% 1) global_rotation 2) scale 3) global_translation
//% 4) wrist
//% 5) thumb1, 6)thumb2, 7) thumb3, 8) thumb4
//%       similarly: index, middle, ring, pinky
//%       end) forearm
void to_pose_params(int count, double const *__restrict theta,
                    const char **__restrict bone_names,
                    Matrix *__restrict pose_params) {
  int i;

  resize(pose_params, 3, count + 3);
  fill(pose_params, 0.0);

  for (i = 0; i < pose_params->nrows; i++) {
    pose_params->data[i] = theta[i];
    pose_params->data[i + 1 * pose_params->nrows] = 1.0;
    pose_params->data[i + 2 * pose_params->nrows] = theta[i + 3];
  }

  int i_theta = 6;
  int i_pose_params = 5;
  int n_fingers = 5;
  int i_finger;
  for (i_finger = 0; i_finger < n_fingers; i_finger++) {
    for (i = 2; i <= 4; i++) {
      pose_params->data[0 + i_pose_params * pose_params->nrows] =
          theta[i_theta];
      i_theta++;

      if (i == 2) {
        pose_params->data[1 + i_pose_params * pose_params->nrows] =
            theta[i_theta];
        i_theta++;
      }

      i_pose_params++;
    }

    i_pose_params++;
  }
}

void hand_objective(double const *__restrict theta, int bone_count,
                    const char **__restrict bone_names,
                    const int *__restrict parents,
                    Matrix *__restrict base_relatives,
                    Matrix *__restrict inverse_base_absolutes,
                    Matrix *__restrict base_positions,
                    Matrix *__restrict weights,
                    const Triangle *__restrict triangles, int is_mirrored,
                    int corresp_count, const int *__restrict correspondences,
                    Matrix *points, double *__restrict err) {
  Matrix *pose_params = get_new_empty_matrix();
  to_pose_params(bone_count, theta, bone_names, pose_params);

  Matrix *vertex_positions = get_new_empty_matrix();
  get_skinned_vertex_positions(bone_count, base_relatives, parents,
                               inverse_base_absolutes, base_positions, weights,
                               is_mirrored, pose_params, vertex_positions, 1);

  int i, j;
  for (i = 0; i < corresp_count; i++) {
    for (j = 0; j < 3; j++) {
      err[i * 3 + j] =
          points->data[j + i * points->nrows] -
          vertex_positions
              ->data[j + correspondences[i] * vertex_positions->nrows];
    }
  }

  delete_matrix(pose_params);
  delete_matrix(vertex_positions);
}

void hand_objective_complicated(
    double const *theta, double const *us, int bone_count,
    const char **bone_names, const int *parents, Matrix *base_relatives,
    Matrix *inverse_base_absolutes, Matrix *base_positions, Matrix *weights,
    const Triangle *triangles, int is_mirrored, int corresp_count,
    const int *__restrict correspondences, Matrix *points, double *err) {
  Matrix *pose_params = get_new_empty_matrix();
  to_pose_params(bone_count, theta, bone_names, pose_params);

  Matrix *vertex_positions = get_new_empty_matrix();
  get_skinned_vertex_positions(bone_count, base_relatives, parents,
                               inverse_base_absolutes, base_positions, weights,
                               is_mirrored, pose_params, vertex_positions, 1);

  int i, j;
  for (i = 0; i < corresp_count; i++) {
    const int *verts = triangles[correspondences[i]].verts;
    double const *u = &us[2 * i];
    for (j = 0; j < 3; j++) {
      double hand_point_coord =
          u[0] *
              vertex_positions->data[j + verts[0] * vertex_positions->nrows] +
          u[1] *
              vertex_positions->data[j + verts[1] * vertex_positions->nrows] +
          (1.0 - u[0] - u[1]) *
              vertex_positions->data[j + verts[2] * vertex_positions->nrows];

      err[i * 3 + j] = points->data[j + i * points->nrows] - hand_point_coord;
    }
  }
}

//*      tapenade -o hand_tapenade -head "hand_objective(err)/(theta)
// hand_objective_complicated(err)/(theta us)" hand.c

extern int enzyme_const;
extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern void __enzyme_autodiff(void *, ...);
void dhand_objective(double const *theta, double *dtheta, int bone_count,
                     const char **bone_names, const int *parents,
                     Matrix *base_relatives, Matrix *inverse_base_absolutes,
                     Matrix *base_positions, Matrix *weights,
                     const Triangle *triangles, int is_mirrored,
                     int corresp_count, const int *correspondences,
                     Matrix *points, double *err, double *derr) {
  __enzyme_autodiff(
      hand_objective, enzyme_dup, theta, dtheta, enzyme_const, bone_count,
      enzyme_const, bone_names, enzyme_const, parents, enzyme_const,
      base_relatives, enzyme_const, inverse_base_absolutes, enzyme_const,
      base_positions, enzyme_const, weights, enzyme_const, triangles,
      enzyme_const, is_mirrored, enzyme_const, corresp_count, enzyme_const,
      correspondences, enzyme_const, points, enzyme_dup, err, derr);
}

void dhand_objective_complicated(double const *theta, double *dtheta,
                                 double const *us, double *dus, int bone_count,
                                 const char **bone_names, const int *parents,
                                 Matrix *base_relatives,
                                 Matrix *inverse_base_absolutes,
                                 Matrix *base_positions, Matrix *weights,
                                 const Triangle *triangles, int is_mirrored,
                                 int corresp_count, const int *correspondences,
                                 Matrix *points, double *err, double *derr) {
  // __enzyme_autodiff(hand_objective_complicated, enzyme_dup, theta, dtheta,
  //                   enzyme_dup, us, dus, enzyme_const, bone_count,
  //                   enzyme_const, bone_names, enzyme_const, parents,
  //                   enzyme_const, base_relatives, enzyme_const,
  //                   inverse_base_absolutes, enzyme_const, base_positions,
  //                   enzyme_const, weights, enzyme_const, triangles,
  //                   enzyme_const, is_mirrored, enzyme_const, corresp_count,
  //                   enzyme_const, correspondences, enzyme_const, points,
  //                   enzyme_dupnoneed, err, derr);
}