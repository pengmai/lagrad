#include "hand.h"
#include "mlir_c_abi.h"
#include <stdio.h>
#include <sys/time.h>

extern void aapointer(double const *, double *);
extern void daapointer(double const *, double *, double *, double *);

extern void
hand_objective(double const *__restrict theta, int bone_count,
               const char **__restrict bone_names,
               const int *__restrict parents, Matrix *__restrict base_relatives,
               Matrix *__restrict inverse_base_absolutes,
               Matrix *__restrict base_positions, Matrix *__restrict weights,
               const Triangle *__restrict triangles, int is_mirrored,
               int corresp_count, const int *__restrict correspondences,
               Matrix *points, double *__restrict err);

extern void mlir_hand_objective(
    /*theta=*/double *, double *, int64_t, int64_t, int64_t,
    /*parents=*/int32_t *, int32_t *, int64_t, int64_t, int64_t,
    /*base_relatives=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    /*inverse_base_absolutes=*/double *, double *, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t,
    /*base_positions=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t,
    /*weights=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
    int64_t);

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
extern F64Descriptor2D dget_posed_relatives(/*base_relatives=*/double *,
                                            double *, int64_t, int64_t, int64_t,
                                            int64_t, int64_t, int64_t, int64_t,
                                            /*pose_params=*/double *, double *,
                                            int64_t, int64_t, int64_t, int64_t,
                                            int64_t);
extern F64Descriptor1D
dtest(/*theta=*/double *, double *, int64_t, int64_t, int64_t,
      /*base_relatives=*/double *, double *, int64_t, int64_t, int64_t, int64_t,
      int64_t, int64_t, int64_t,
      /*parents=*/int32_t *, int32_t *, int64_t, int64_t, int64_t);
extern F64Descriptor3D mrelatives_to_absolutes(/*relatives=*/double *, double *,
                                               int64_t, int64_t, int64_t,
                                               int64_t, int64_t, int64_t,
                                               int64_t, /*parents=*/int32_t *,
                                               int32_t *, int64_t, int64_t,
                                               int64_t);
// extern F64Descriptor1D mget_skinned_vertex_positions(
//     /*base_relatives=*/double *, double *, int64_t, int64_t, int64_t,
//     int64_t, int64_t, int64_t, int64_t, /*parents=*/int *, int *, int64_t,
//     int64_t, int64_t, /**/
// );
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

unsigned long enzyme_colmaj_hand_simple(HandInput *input,
                                        struct MatrixConverted *converted,
                                        double *J, double *ref_J) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  enzyme_jacobian_simple(
      input, converted->base_relatives, converted->inverse_base_absolutes,
      converted->base_positions, converted->weights, converted->points, J);
  gettimeofday(&stop, NULL);
  double err = 0.0;
  for (size_t i = 0; i < 3 * input->n_pts * input->n_theta; i++) {
    err += fabs(J[i]);
  }
  if (err < 1e-6) {
    printf("Enzyme AD was disabled\n");
  } else {
    verify_hand_results(ref_J, J, 3 * input->n_pts, input->n_theta,
                        "Enzyme Simple Column Major/C");
  }
  return timediff(start, stop);
}

int main() {
  double a[3] = {1., 2., 3.};
  double da[3] = {0};
  double R[9] = {0};
  double dR[9] = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
  // aapointer(a, R);
  daapointer(a, da, R, dR);
  print_d_arr(da, 3);
  print_d_arr_2d(R, 3, 3);
  return 0;
  /* Preamble */
  HandInput input = read_hand_data(false, true);

  Matrix *base_relatives =
      ptr_to_matrices(input.model.base_relatives, input.model.n_bones, 4, 4);
  Matrix *inverse_base_absolutes = ptr_to_matrices(
      input.model.inverse_base_absolutes, input.model.n_bones, 4, 4);
  Matrix base_positions =
      ptr_to_matrix(input.model.base_positions, 4, input.model.n_vertices);
  Matrix weights = ptr_to_matrix(input.model.weights, input.model.n_bones,
                                 input.model.n_vertices);
  Matrix points = ptr_to_matrix(input.points, 3, input.n_pts);
  // struct MatrixConverted converted = {.base_relatives = base_relatives,
  //                                     .inverse_base_absolutes =
  //                                         inverse_base_absolutes,
  //                                     .base_positions = &base_positions,
  //                                     .weights = &weights,
  //                                     .points = &points};
  int J_rows = 3 * input.n_pts;
  double *ref_J = (double *)malloc(J_rows * input.n_theta * sizeof(double));
  parse_hand_results("benchmarks/results/hand_test.txt", ref_J, J_rows,
                     input.n_theta);
  double *J = (double *)malloc(J_rows * input.n_theta * sizeof(double));

  // unsigned long ecm = enzyme_colmaj_hand_simple(&input, &converted, J,
  // ref_J); printf("Enzyme colmajor took: %lu\n", ecm);

  double err[3 * input.n_pts];
  hand_objective(input.theta, input.model.n_bones, input.model.bone_names,
                 input.model.parents, base_relatives, inverse_base_absolutes,
                 &base_positions, &weights, NULL, input.model.is_mirrored,
                 input.n_pts, input.correspondences, &points, err);
  // print_d_arr(err, 6);

  double *deadbeef = (double *)0xdeadbeef;
  // print_d_arr(input.correspondences, 6);
  mlir_hand_objective(
      /*theta=*/deadbeef, input.theta, 0, input.n_theta, 1,
      /*parents=*/(int32_t *)deadbeef, input.model.parents, 0,
      input.model.n_bones, 1,
      /*base_relatives=*/deadbeef, input.model.base_relatives, 0,
      input.model.n_bones, 4, 4, 16, 4, 1,
      /*inverse_base_absolutes=*/deadbeef, input.model.inverse_base_absolutes,
      0, input.model.n_bones, 4, 4, 16, 4, 1,
      /*base_positions=*/deadbeef, input.model.base_positions, 0,
      input.model.n_vertices, 4, 4, 1,
      /*weights=*/deadbeef, input.model.weights, 0, input.model.n_vertices,
      input.model.n_bones, input.model.n_bones, 1);
  // F64Descriptor2D pose_params =
  //     mto_pose_params(deadbeef, input.theta, 0, input.n_theta, 1);
  // printf("Pose params:\n");
  // print_d_arr_2d(pose_params.aligned, pose_params.size_0,
  // pose_params.size_1);

  // F64Descriptor1D dt =
  //     dtest(deadbeef, input.theta, 0, input.n_theta, 1, deadbeef,
  //           input.model.base_relatives, 0, input.model.n_bones, 4, 4, 16, 4,
  //           1, (int *)deadbeef, input.model.parents, 0, input.model.n_bones,
  //           1);
  // printf("dt:\n");
  // print_d_arr(dt.aligned, dt.size);

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
  free(ref_J);
  free_hand_input(&input);
}
