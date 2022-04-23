/*
A version of hand tracking designed to use the same memory layout
as the MLIR version.
*/

#define TARGET_OS_EMBEDDED 0
#define N_BONES 22
#define N_VERTS 544
#include "shared_types.h"
#include <math.h>
#include <stdlib.h>

/* Row-major matrix multiplication */
void matmul(int size, double *A, double *B, double *C) {
  for (int i = 0; i < size * size; i++) {
    C[i] = 0;
  }

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        C[i * size + j] += A[i * size + k] * B[k * size + j];
      }
    }
  }
}

/* PM * NP -> NM */
void colmaj_matmul(int m, int n, int p, double *A, double *B, double *C) {
  for (int i = 0; i < n * m; i++) {
    C[i] = 0;
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        C[j * m + i] += A[k * p + i] * B[j * m + k];
      }
    }
  }
}

void cto_pose_params(double *theta, double *pose_params) {
  for (int i = 0; i < 25 * 3; i++) {
    pose_params[i] = 0;
  }

  for (int i = 0; i < 3; i++) {
    pose_params[0 * 3 + i] = theta[i];
    pose_params[1 * 3 + i] = 1.0;
    pose_params[2 * 3 + i] = theta[i + 3];
  }

  int theta_idx = 6;
  int pose_idx = 5;
  for (int i = 0; i < 5; i++) {
    pose_params[pose_idx * 3 + 0] = theta[theta_idx];

    theta_idx++;
    pose_params[pose_idx * 3 + 1] = theta[theta_idx];

    theta_idx++;
    pose_idx++;
    pose_params[pose_idx * 3 + 0] = theta[theta_idx];

    theta_idx++;
    pose_idx++;
    pose_params[pose_idx * 3 + 0] = theta[theta_idx];

    pose_idx += 2;
    theta_idx++;
  }
}

void ceuler_angles_to_rotation_matrix(double *xzy, double *R, int print) {
  double tx = xzy[0];
  double ty = xzy[2];
  double tz = xzy[1];
  double Rx[9];
  for (int i = 0; i < 9; i++) {
    Rx[i] = 0;
  }
  Rx[1 * 3 + 1] = cos(tx);
  Rx[1 * 3 + 2] = sin(tx);
  Rx[2 * 3 + 1] = -Rx[1 * 3 + 2];
  Rx[2 * 3 + 2] = Rx[1 * 3 + 1];
  Rx[0 * 3 + 0] = 1;

  if (print) {
    // printf("Rx:\n");
    // print_d_arr_2d(Rx, 3, 3);
  }

  double Ry[9];
  for (int i = 0; i < 9; i++) {
    Ry[i] = 0;
  }
  Ry[0 * 3 + 0] = cos(ty);
  Ry[0 * 3 + 2] = sin(ty);
  Ry[2 * 3 + 0] = -Ry[0 * 3 + 2];
  Ry[2 * 3 + 2] = Ry[0 * 3 + 0];
  Ry[1 * 3 + 1] = 1;

  double Rz[9];
  for (int i = 0; i < 9; i++) {
    Rz[i] = 0;
  }
  Rz[0 * 3 + 0] = cos(tz);
  Rz[0 * 3 + 1] = sin(tz);
  Rz[1 * 3 + 0] = -Rz[0 * 3 + 1];
  Rz[1 * 3 + 1] = Rz[0 * 3 + 0];
  Rz[2 * 3 + 2] = 1;

  double tmp[9];
  matmul(3, Ry, Rz, tmp);
  matmul(3, Rx, tmp, R);
}

void cget_posed_relatives(double *base_relatives, double *pose_params,
                          double *relatives) {
  double tr_space[16];
  for (size_t i = 0; i < 4 * 4; i++) {
    tr_space[i] = 0;
  }

  double R[9];
  for (int i = 0; i < N_BONES; i++) {
    ceuler_angles_to_rotation_matrix(&pose_params[(i + 3) * 3], R, i == 6);
    // if (i == 6) {
    //   print_d_arr_2d(R, 3, 3);
    // }
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        tr_space[j * 4 + k] = R[j * 3 + k];
      }
    }
    tr_space[3 * 4 + 3] = 1;

    matmul(4, tr_space, &base_relatives[i * 4 * 4], &relatives[i * 4 * 4]);
  }
}

void crelatives_to_absolutes(double *relatives, int32_t *parents,
                             double *absolutes) {
  for (int i = 0; i < N_BONES; i++) {
    if (parents[i] == -1) {
      // copy
      for (int j = 0; j < 4 * 4; j++) {
        absolutes[i * 4 * 4 + j] = relatives[i * 4 * 4 + j];
      }
    } else {
      matmul(4, &relatives[i * 4 * 4], &absolutes[parents[i] * 4 * 4],
             &absolutes[i * 4 * 4]);
    }
  }
}

void cangle_axis_to_rotation_matrix(double *angle_axis, double *R) {
  // square sum
  double norm = 0;
  for (int i = 0; i < 3; i++) {
    norm += angle_axis[i] * angle_axis[i];
  }
  norm = sqrt(norm);
  if (norm < 0.0001) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        R[i * 3 + j] = i == j ? 1.0 : 0.0;
      }
    }
    return;
  }

  double x = angle_axis[0] / norm;
  double y = angle_axis[1] / norm;
  double z = angle_axis[2] / norm;

  double s = sin(norm);
  double c = cos(norm);

  // First row
  R[0 * 3 + 0] = x * x + (1 - x * x) * c;
  R[0 * 3 + 1] = x * y * (1 - c) + z * s;
  R[0 * 3 + 2] = x * z * (1 - c) - y * s;

  // Second row
  R[1 * 3 + 0] = x * y * (1 - c) - z * s;
  R[1 * 3 + 1] = y * y + (1 - y * y) * c;
  R[1 * 3 + 2] = z * y * (1 - c) + x * s;

  // Third row
  R[2 * 3 + 0] = x * z * (1 - c) + y * s;
  R[2 * 3 + 1] = y * z * (1 - c) - x * s;
  R[2 * 3 + 2] = z * z + (1 - z * z) * c;
}

void capply_global_transforms(double *pose_params, double *positions) {
  double *R = malloc(3 * 3 * sizeof(double));
  cangle_axis_to_rotation_matrix(pose_params, R);

  // printf("colmaj R:\n");
  // print_d_arr_2d(R, 3, 3);
  // for (int i = 0; i < 3; i++) {
  //   for (int j = 0; j < 3; j++) {
  //     R[i * 3 + j] *= pose_params[1 * 3 + i];
  //   }
  // }

  double *tmp = malloc(N_VERTS * 3 * sizeof(double));
  colmaj_matmul(3, N_VERTS, 3, R, positions, tmp);

  for (int i = 0; i < N_VERTS; i++) {
    for (int j = 0; j < 3; j++) {
      positions[i * 3 + j] = tmp[i * 3 + j] + pose_params[2 * 3 + j];
    }
  }

  free(R);
  free(tmp);
}

void c_packed_hand_objective(int npts, double const *__restrict theta,
                             int32_t *parents,
                             double const *__restrict base_relatives,
                             double const *__restrict inverse_base_absolutes,
                             double const *__restrict base_positions,
                             double const *__restrict weights,
                             int32_t *correspondences, double *points,
                             double *err) {
  double *pose_params = malloc(25 * 3 * sizeof(double));
  double *relatives = malloc(N_BONES * 4 * 4 * sizeof(double));
  double *absolutes = malloc(N_BONES * 4 * 4 * sizeof(double));
  double *transforms = malloc(N_BONES * 4 * 4 * sizeof(double));
  double *curr_positions = malloc(N_VERTS * 4 * sizeof(double));
  double *positions = malloc(N_VERTS * 3 * sizeof(double));
  cto_pose_params(theta, pose_params);
  cget_posed_relatives(base_relatives, pose_params, relatives);
  crelatives_to_absolutes(relatives, parents, absolutes);

  for (int i = 0; i < N_BONES; i++) {
    matmul(4, &inverse_base_absolutes[i * 4 * 4], &absolutes[i * 4 * 4],
           &transforms[i * 4 * 4]);
  }

  for (int i = 0; i < N_VERTS * 3; i++) {
    positions[i] = 0;
  }

  for (int i = 0; i < N_BONES; i++) {
    colmaj_matmul(4, N_VERTS, 4, &transforms[i * 4 * 4], base_positions,
                  curr_positions);

    for (int j = 0; j < N_VERTS; j++) {
      for (int k = 0; k < 3; k++) {
        positions[j * 3 + k] +=
            curr_positions[j * 4 + k] * weights[j * N_BONES + i];
      }
    }
  }

  /* Apply global transforms */
  capply_global_transforms(pose_params, positions);

  for (int i = 0; i < npts; i++) {
    for (int j = 0; j < 3; j++) {
      err[i * 3 + j] =
          points[i * 3 + j] - positions[correspondences[i] * 3 + j];
    }
  }

  printf("err:\n");
  print_d_arr_2d(&err[0], 10, 3);

  free(pose_params);
  free(relatives);
  free(absolutes);
  free(transforms);
  free(curr_positions);
  free(positions);
}