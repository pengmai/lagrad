#include "hand.h"
#include "mlir_c_abi.h"
#include <stdio.h>
#include <sys/time.h>

extern void
hand_objective(double const *__restrict theta, int bone_count,
               const char **__restrict bone_names,
               const int *__restrict parents, Matrix *__restrict base_relatives,
               Matrix *__restrict inverse_base_absolutes,
               Matrix *__restrict base_positions, Matrix *__restrict weights,
               const Triangle *__restrict triangles, int is_mirrored,
               int corresp_count, const int *__restrict correspondences,
               Matrix *points, double *__restrict err);
extern void dhand_objective(double const *theta, double *dtheta, int bone_count,
                            const char **bone_names, const int *parents,
                            Matrix *base_relatives,
                            Matrix *inverse_base_absolutes,
                            Matrix *base_positions, Matrix *weights,
                            const Triangle *triangles, int is_mirrored,
                            int corresp_count, const int *correspondences,
                            Matrix *points, double *err, double *derr);

/* For debugging */
extern F64Descriptor2D mto_pose_params(/*theta=*/double *, double *, int64_t,
                                       int64_t, int64_t);
extern F64Descriptor1D dtopose_params(/*theta=*/double *, double *, int64_t,
                                      int64_t, int64_t);
extern F64Descriptor3D mget_posed_relatives(/*base_relatives=*/double *,
                                            double *, int64_t, int64_t, int64_t,
                                            int64_t, int64_t, int64_t, int64_t,
                                            /*pose_params=*/double *, double *,
                                            int64_t, int64_t, int64_t, int64_t,
                                            int64_t);
// TODO: This is incorrect
extern F64Descriptor2D dget_posed_relatives(/*base_relatives=*/double *,
                                            double *, int64_t, int64_t, int64_t,
                                            int64_t, int64_t, int64_t, int64_t,
                                            /*pose_params=*/double *, double *,
                                            int64_t, int64_t, int64_t, int64_t,
                                            int64_t);
extern F64Descriptor1D dtest(/*theta=*/double *, double *, int64_t, int64_t,
                             int64_t,
                             /*base_relatives=*/double *, double *, int64_t,
                             int64_t, int64_t, int64_t, int64_t, int64_t,
                             int64_t);
void enzyme_jacobian_simple(HandInput *input, Matrix *base_relatives,
                            Matrix *inverse_base_absolutes,
                            Matrix *base_positions, Matrix *weights,
                            Matrix *points, double *J) {
  int err_size = 3 * input->n_pts;
  for (size_t i = 0; i < err_size; i++) {
    double *dtheta = (double *)malloc(input->n_theta * sizeof(double));
    for (size_t j = 0; j < input->n_theta; j++) {
      dtheta[j] = 0;
    }

    double *err = (double *)malloc(err_size * sizeof(double));
    double *derr = (double *)malloc(err_size * sizeof(double));
    for (size_t j = 0; j < err_size; j++) {
      err[j] = 0;
      derr[j] = (i == j) ? 1.0 : 0.0;
    }

    dhand_objective(input->theta, dtheta, input->model.n_bones,
                    input->model.bone_names, input->model.parents,
                    base_relatives, inverse_base_absolutes, base_positions,
                    weights, NULL, input->model.is_mirrored, input->n_pts,
                    input->correspondences, points, err, derr);
    for (size_t j = 0; j < input->n_theta; j++) {
      J[i * input->n_theta + j] = dtheta[j];
    }

    free(dtheta);
    free(err);
    free(derr);
  }
}

int main() {
  /* Preamble */
  HandInput input = read_hand_data(false);

  Matrix *base_relatives =
      ptr_to_matrices(input.model.base_relatives, input.model.n_bones, 4, 4);
  Matrix *inverse_base_absolutes = ptr_to_matrices(
      input.model.inverse_base_absolutes, input.model.n_bones, 4, 4);
  Matrix base_positions =
      ptr_to_matrix(input.model.base_positions, 4, input.model.n_vertices);
  Matrix weights = ptr_to_matrix(input.model.weights, input.model.n_bones,
                                 input.model.n_vertices);
  Matrix points = ptr_to_matrix(input.points, 3, input.n_pts);
  int J_rows = 3 * input.n_pts;
  // double *ref_J = (double *)malloc(J_rows * input.n_theta * sizeof(double));
  // parse_hand_results("benchmarks/results/hand_test.txt", ref_J, J_rows,
  //                    input.n_theta);
  double *J = (double *)malloc(J_rows * input.n_theta * sizeof(double));

  double err[3 * input.n_pts];
  // enzyme_jacobian_simple(&input, base_relatives, inverse_base_absolutes,
  //                        &base_positions, &weights, &points, J);
  hand_objective(input.theta, input.model.n_bones, input.model.bone_names,
                 input.model.parents, base_relatives, inverse_base_absolutes,
                 &base_positions, &weights, NULL, input.model.is_mirrored,
                 input.n_pts, input.correspondences, &points, err);

  double *deadbeef = (double *)0xdeadbeef;
  F64Descriptor2D pose_params =
      mto_pose_params(deadbeef, input.theta, 0, input.n_theta, 1);
  // printf("Pose params:\n");
  // print_d_arr_2d(pose_params.aligned, pose_params.size_0,
  // pose_params.size_1);

  // F64Descriptor1D dpp =
  //     dtopose_params(deadbeef, input.theta, 0, input.n_theta, 1);
  // printf("dPose params:\n");
  // print_d_arr(dpp.aligned, dpp.size);

  // F64Descriptor3D relatives = mget_posed_relatives(
  //     deadbeef, input.model.base_relatives, 0, input.model.n_bones, 4, 4, 16,
  //     4, 1, pose_params.allocated, pose_params.aligned, pose_params.offset,
  //     pose_params.size_0, pose_params.size_1, pose_params.stride_0,
  //     pose_params.stride_1);

  F64Descriptor1D dt =
      dtest(deadbeef, input.theta, 0, input.n_theta, 1, deadbeef,
            input.model.base_relatives, 0, input.model.n_bones, 4, 4, 16, 4, 1);
  printf("dt:\n");
  print_d_arr(dt.aligned, dt.size);

  // F64Descriptor2D drelatives = dget_posed_relatives(
  //     deadbeef, input.model.base_relatives, 0, input.model.n_bones, 4, 4, 16,
  //     4, 1, pose_params.allocated, pose_params.aligned, pose_params.offset,
  //     pose_params.size_0, pose_params.size_1, pose_params.stride_0,
  //     pose_params.stride_1);
  // printf("drelatives:\n");
  // print_d_arr_2d(drelatives.aligned, drelatives.size_0, drelatives.size_1);
  // verify_hand_results(ref_J, J, J_rows, input.n_theta, "Enzyme/C");
  // serialize_hand_results("benchmarks/results/hand_test.txt", J, J_rows,
  //                        input.n_theta);

  /* Cleanup */
  free_matrix_array(base_relatives, input.model.n_bones);
  free_matrix_array(inverse_base_absolutes, input.model.n_bones);
  free(base_positions.data);
  free(weights.data);
  free(points.data);
  free(J);
}
