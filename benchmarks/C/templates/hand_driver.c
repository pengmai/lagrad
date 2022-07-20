#include "hand.h"
#include "memusage.h"
#include "mlir_c_abi.h"
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#define NUM_RUNS 6
int CHECK_MEM = 0;
double *deadbeef = (double *)0xdeadbeef;
RunProcDyn rpd;
void check_mem_usage() {
  run_get_dynamic_proc_info(getpid(), &rpd);
  printf("%zu\t%zu\n", rpd.rss, rpd.vsize);
}

extern void
hand_objective(double const *__restrict theta, int bone_count,
               const char **__restrict bone_names,
               const int *__restrict parents, Matrix *__restrict base_relatives,
               Matrix *__restrict inverse_base_absolutes,
               Matrix *__restrict base_positions, Matrix *__restrict weights,
               const Triangle *__restrict triangles, int is_mirrored,
               int corresp_count, const int *__restrict correspondences,
               Matrix *points, double *__restrict err);

extern F64Descriptor2D mlir_hand_objective(
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
    /*points=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);

extern double emlir_hand_objective(
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
    /*out=*/double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t);

extern F64Descriptor1D lagrad_hand_objective(
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

typedef struct HandComplicatedGrad {
  F64Descriptor1D dtheta;
  F64Descriptor2D dus;
} HandComplicatedGrad;

extern HandComplicatedGrad lagrad_hand_objective_complicated(
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

extern F64Descriptor1D enzyme_hand_objective(
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

extern HandComplicatedGrad enzyme_hand_objective_complicated(
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

extern void dhand_objective(double const *theta, double *dtheta, int bone_count,
                            const char **bone_names, const int *parents,
                            Matrix *base_relatives,
                            Matrix *inverse_base_absolutes,
                            Matrix *base_positions, Matrix *weights,
                            const Triangle *triangles, int is_mirrored,
                            int corresp_count, const int *correspondences,
                            Matrix *points, double *err, double *derr);

extern void dhand_objective_complicated(
    double const *theta, double *dtheta, double const *us, double *dus,
    int bone_count, const char **bone_names, const int *parents,
    Matrix *base_relatives, Matrix *inverse_base_absolutes,
    Matrix *base_positions, Matrix *weights, const Triangle *triangles,
    int is_mirrored, int corresp_count, const int *correspondences,
    Matrix *points, double *err, double *derr);

extern void enzyme_c_packed_hand_objective(
    int npts, double const *__restrict theta, double *dtheta, int32_t *parents,
    double const *__restrict base_relatives,
    double const *__restrict inverse_base_absolutes,
    double const *__restrict base_positions, double const *__restrict weights,
    int32_t *correspondences, double *points, double *err, double *derr);

void enzyme_c_jacobian_simple(HandInput *input, Matrix *base_relatives,
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

void enzyme_c_jacobian_complicated(HandInput *input, Matrix *base_relatives,
                                   Matrix *inverse_base_absolutes,
                                   Matrix *base_positions, Matrix *weights,
                                   Triangle *triangles, Matrix *points,
                                   double *J) {
  int err_size = 3 * input->n_pts;
  int J_stride = input->n_theta + 2;
  double *err = malloc(err_size * sizeof(double));
  double *derr = malloc(err_size * sizeof(double));
  for (size_t i = 0; i < err_size; i++) {
    double *dtheta = calloc(input->n_theta, sizeof(double));
    double *dus = calloc(input->n_pts * 2, sizeof(double));
    for (size_t j = 0; j < err_size; j++) {
      derr[j] = (i == j) ? 1.0 : 0.0;
    }
    dhand_objective_complicated(
        input->theta, dtheta, input->us, dus, input->model.n_bones,
        input->model.bone_names, input->model.parents, base_relatives,
        inverse_base_absolutes, base_positions, weights, triangles,
        input->model.is_mirrored, input->n_pts, input->correspondences, points,
        err, derr);

    // Write theta part
    for (size_t j = 0; j < input->n_theta; j++) {
      J[i * J_stride + 2 + j] = dtheta[j];
    }
    // Write us part
    for (size_t j = 0; j < 2; j++) {
      J[i * J_stride + j] = dus[(i / 3) * 2 + j];
    }
    free(dtheta);
    free(dus);
  }

  free(err);
  free(derr);
}

void enzyme_c_packed_jacobian_simple(HandInput *input, double *J) {
  int err_size = 3 * input->n_pts;
  double *err = (double *)malloc(err_size * sizeof(double));
  double *derr = (double *)malloc(err_size * sizeof(double));
  for (size_t i = 0; i < err_size; i++) {
    double *dtheta = (double *)calloc(input->n_theta, sizeof(double));

    for (size_t j = 0; j < err_size; j++) {
      err[j] = 0;
      derr[j] = (i == j) ? 1.0 : 0.0;
    }

    enzyme_c_packed_hand_objective(
        input->n_pts, input->theta, dtheta, input->model.parents,
        input->model.base_relatives, input->model.inverse_base_absolutes,
        input->model.base_positions, input->model.weights,
        input->correspondences, input->points, err, derr);

    for (size_t j = 0; j < input->n_theta; j++) {
      J[i * input->n_theta + j] = dtheta[j];
    }

    free(dtheta);
  }
  free(err);
  free(derr);
}

void lagrad_jacobian_simple(HandInput *input, double *J) {
  int err_size = 3 * input->n_pts;
  for (size_t i = 0; i < err_size; i++) {
    double *derr = (double *)malloc(err_size * sizeof(double));
    for (size_t j = 0; j < err_size; j++) {
      derr[j] = (i == j) ? 1.0 : 0.0;
    }

    F64Descriptor1D dtheta = lagrad_hand_objective(
        /*theta=*/deadbeef, input->theta, 0, input->n_theta, 1,
        /*parents=*/(int32_t *)deadbeef, input->model.parents, 0,
        input->model.n_bones, 1,
        /*base_relatives=*/deadbeef, input->model.base_relatives, 0,
        input->model.n_bones, 4, 4, 16, 4, 1,
        /*inverse_base_absolutes=*/deadbeef,
        input->model.inverse_base_absolutes, 0, input->model.n_bones, 4, 4, 16,
        4, 1,
        /*base_positions=*/deadbeef, input->model.base_positions, 0,
        input->model.n_vertices, 4, 4, 1,
        /*weights=*/deadbeef, input->model.weights, 0, input->model.n_vertices,
        input->model.n_bones, input->model.n_bones, 1,
        /*correspondences=*/(int32_t *)deadbeef, input->correspondences, 0,
        input->n_pts, 1,
        /*points=*/deadbeef, input->points, 0, input->n_pts, 3, 3, 1,
        /*g=*/deadbeef, derr, 0, input->n_pts, 3, 3, 1);
    for (size_t j = 0; j < input->n_theta; j++) {
      J[i * input->n_theta + j] = dtheta.aligned[j];
    }
    free(dtheta.aligned);
    free(derr);
  }
}

void lagrad_jacobian_complicated(HandInput *input, double *J) {
  int err_size = 3 * input->n_pts;
  int J_stride = input->n_theta + 2;
  double *derr = malloc(err_size * sizeof(double));
  for (size_t i = 0; i < err_size; i++) {
    for (size_t j = 0; j < err_size; j++) {
      derr[j] = (i == j) ? 1.0 : 0.0;
    }
    HandComplicatedGrad res = lagrad_hand_objective_complicated(
        /*theta=*/deadbeef, input->theta, 0, input->n_theta, 1,
        /*us=*/deadbeef, input->us, 0, input->n_pts, 2, 2, 1,
        /*parents=*/(int32_t *)deadbeef, input->model.parents, 0,
        input->model.n_bones, 1,
        /*base_relatives=*/deadbeef, input->model.base_relatives, 0,
        input->model.n_bones, 4, 4, 16, 4, 1,
        /*inverse_base_absolutes=*/deadbeef,
        input->model.inverse_base_absolutes, 0, input->model.n_bones, 4, 4, 16,
        4, 1,
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

    // Write theta part
    for (size_t j = 0; j < res.dtheta.size; j++) {
      J[i * J_stride + 2 + j] = res.dtheta.aligned[j];
    }
    // Write us part
    for (size_t j = 0; j < 2; j++) {
      J[i * J_stride + j] = res.dus.aligned[(i / 3) * 2 + j];
    }
    free(res.dtheta.aligned);
    free(res.dus.aligned);
  }
  free(derr);
}

void enzyme_jacobian_simple(HandInput *input, double *J) {
  int err_size = 3 * input->n_pts;
  for (size_t i = 0; i < err_size; i++) {
    double *derr = (double *)malloc(err_size * sizeof(double));
    for (size_t j = 0; j < err_size; j++) {
      derr[j] = (i == j) ? 1.0 : 0.0;
    }
    F64Descriptor1D dtheta = enzyme_hand_objective(
        /*theta=*/deadbeef, input->theta, 0, input->n_theta, 1,
        /*parents=*/(int32_t *)deadbeef, input->model.parents, 0,
        input->model.n_bones, 1,
        /*base_relatives=*/deadbeef, input->model.base_relatives, 0,
        input->model.n_bones, 4, 4, 16, 4, 1,
        /*inverse_base_absolutes=*/deadbeef,
        input->model.inverse_base_absolutes, 0, input->model.n_bones, 4, 4, 16,
        4, 1,
        /*base_positions=*/deadbeef, input->model.base_positions, 0,
        input->model.n_vertices, 4, 4, 1,
        /*weights=*/deadbeef, input->model.weights, 0, input->model.n_vertices,
        input->model.n_bones, input->model.n_bones, 1,
        /*correspondences=*/(int32_t *)deadbeef, input->correspondences, 0,
        input->n_pts, 1,
        /*points=*/deadbeef, input->points, 0, input->n_pts, 3, 3, 1,
        /*g=*/deadbeef, derr, 0, input->n_pts, 3, 3, 1);
    for (size_t j = 0; j < input->n_theta; j++) {
      J[i * input->n_theta + j] = dtheta.aligned[j];
    }
    free(dtheta.aligned);
    free(derr);
  }
}

void enzyme_mlir_jacobian_complicated(HandInput *input, double *J) {
  int err_size = 3 * input->n_pts;
  int J_stride = input->n_theta + 2;
  double *derr = malloc(err_size * sizeof(double));
  for (size_t i = 0; i < err_size; i++) {
    for (size_t j = 0; j < err_size; j++) {
      derr[j] = (i == j) ? 1.0 : 0.0;
    }
    HandComplicatedGrad res = enzyme_hand_objective_complicated(
        /*theta=*/deadbeef, input->theta, 0, input->n_theta, 1,
        /*us=*/deadbeef, input->us, 0, input->n_pts, 2, 2, 1,
        /*parents=*/(int32_t *)deadbeef, input->model.parents, 0,
        input->model.n_bones, 1,
        /*base_relatives=*/deadbeef, input->model.base_relatives, 0,
        input->model.n_bones, 4, 4, 16, 4, 1,
        /*inverse_base_absolutes=*/deadbeef,
        input->model.inverse_base_absolutes, 0, input->model.n_bones, 4, 4, 16,
        4, 1,
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

    // Write theta part
    for (size_t j = 0; j < res.dtheta.size; j++) {
      J[i * J_stride + 2 + j] = res.dtheta.aligned[j];
    }
    // Write us part
    for (size_t j = 0; j < 2; j++) {
      J[i * J_stride + j] = res.dus.aligned[(i / 3) * 2 + j];
    }
    free(res.dtheta.aligned);
    free(res.dus.aligned);
  }
  free(derr);
}

unsigned long enzyme_C_hand_simple(HandInput *input,
                                   struct MatrixConverted *converted, double *J,
                                   double *ref_J) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  enzyme_c_jacobian_simple(
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

unsigned long enzyme_C_hand_complicated(HandInput *input,
                                        struct MatrixConverted *converted,
                                        double *J, double *ref_J) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  enzyme_c_jacobian_complicated(input, converted->base_relatives,
                                converted->inverse_base_absolutes,
                                converted->base_positions, converted->weights,
                                converted->triangles, converted->points, J);
  gettimeofday(&stop, NULL);
  double err = 0.0;
  for (size_t i = 0; i < 3 * input->n_pts * input->n_theta; i++) {
    err += fabs(J[i]);
  }
  if (err < 1e-6) {
    printf("Enzyme AD was disabled\n");
  } else {
    verify_hand_results(ref_J, J, 3 * input->n_pts, input->n_theta,
                        "Enzyme Complicated Column Major/C");
  }
  return timediff(start, stop);
}

typedef unsigned long (*bodyFunc)(HandInput *input, double *J, double *ref_J);

unsigned long lagrad_hand_simple(HandInput *input, double *J, double *ref_J) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  lagrad_jacobian_simple(input, J);
  gettimeofday(&stop, NULL);
  verify_hand_results(ref_J, J, 3 * input->n_pts, input->n_theta,
                      "LAGrad Simple");
  return timediff(start, stop);
}

unsigned long lagrad_hand_complicated(HandInput *input, double *J,
                                      double *ref_J) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  lagrad_jacobian_complicated(input, J);
  gettimeofday(&stop, NULL);
  if (CHECK_MEM) {
    check_mem_usage();
  } else {
    verify_hand_results(ref_J, J, 3 * input->n_pts, 2 + input->n_theta,
                        "LAGrad Complicated");
  }
  return timediff(start, stop);
}

unsigned long enzyme_hand_simple(HandInput *input, double *J, double *ref_J) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  enzyme_jacobian_simple(input, J);
  gettimeofday(&stop, NULL);
  verify_hand_results(ref_J, J, 3 * input->n_pts, input->n_theta,
                      "Enzyme/MLIR");
  return timediff(start, stop);
}

unsigned long enzyme_mlir_hand_complicated(HandInput *input, double *J,
                                           double *ref_J) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  enzyme_mlir_jacobian_complicated(input, J);
  gettimeofday(&stop, NULL);
  if (CHECK_MEM) {
    check_mem_usage();
  } else {
    verify_hand_results(ref_J, J, 3 * input->n_pts, 2 + input->n_theta,
                        "Enzyme/MLIR Complicated");
  }
  return timediff(start, stop);
}

unsigned long collect_enzyme_c_packed_simple(HandInput *input, double *J,
                                             double *ref_J) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  enzyme_c_packed_jacobian_simple(input, J);
  gettimeofday(&stop, NULL);
  if (CHECK_MEM) {
    check_mem_usage();
  } else {
    verify_hand_results(ref_J, J, 3 * input->n_pts, input->n_theta,
                        "Enzyme/C Packed");
    print_d_arr_2d(J, 3 * input->n_pts, input->n_theta);
  }
  return timediff(start, stop);
}

void populate_ref(HandInput *hand_input, double *ref_J, int complicated) {
  if (complicated) {
    lagrad_jacobian_complicated(hand_input, ref_J);
  } else {
    lagrad_jacobian_simple(hand_input, ref_J);
  }
}

/*Testing the packed C primal*/
void c_packed_hand_objective(int npts, double const *__restrict theta,
                             int32_t *parents,
                             double const *__restrict base_relatives,
                             double const *__restrict inverse_base_absolutes,
                             double const *__restrict base_positions,
                             double const *__restrict weights,
                             int32_t *correspondences, double *points,
                             double *err);
int check_main() {
  HandInput hand_input = read_hand_data("{{model_dir}}", "{{data_file}}",
                                        /*complicated=*/0, /*transposed=*/1);
  double *err = malloc(hand_input.n_pts * 3 * sizeof(double));
  c_packed_hand_objective(
      hand_input.n_pts, hand_input.theta, hand_input.model.parents,
      hand_input.model.base_relatives, hand_input.model.inverse_base_absolutes,
      hand_input.model.base_positions, hand_input.model.weights,
      hand_input.correspondences, hand_input.points, err);
  HandInput *input = &hand_input;
  F64Descriptor2D merr = mlir_hand_objective(
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
      /*points=*/deadbeef, input->points, 0, input->n_pts, 3, 3, 1);
  double discrep = 0;
  for (size_t i = 0; i < hand_input.n_pts * 3; i++) {
    discrep += fabs(err[i] - merr.aligned[i]);
  }

  printf("Total error between MLIR and C packed: %.4e\n", discrep);
  double *g = calloc(hand_input.n_pts * 3, sizeof(double));
  for (size_t i = 0; i < hand_input.n_pts * 3; i++) {
    err[i] = 0.0;
    // g[i] = 1.0;
  }
  g[0] = 1.0;
  double *dtheta = calloc(hand_input.n_theta, sizeof(double));
  enzyme_c_packed_hand_objective(
      hand_input.n_pts, hand_input.theta, dtheta, hand_input.model.parents,
      hand_input.model.base_relatives, hand_input.model.inverse_base_absolutes,
      hand_input.model.base_positions, hand_input.model.weights,
      hand_input.correspondences, hand_input.points, err, g);
  printf("C packed Jacobian first row:\n");
  print_d_arr(dtheta, hand_input.n_theta);
  free(dtheta);
  free(g);
  return 0;
}

extern void hand_objective_complicated(
    double const *theta, double const *us, int bone_count,
    const char **bone_names, const int *parents, Matrix *base_relatives,
    Matrix *inverse_base_absolutes, Matrix *base_positions, Matrix *weights,
    const Triangle *triangles, int is_mirrored, int corresp_count,
    const int *__restrict correspondences, Matrix *points, double *err);

int main() {
  /* Preamble */
  CHECK_MEM = strtol("{{measure_mem|int}}", NULL, 10);
  int complicated = strtol("{{complicated|int}}", NULL, 10);
  HandInput input =
      read_hand_data("{{model_dir}}", "{{data_file}}",
                     /*complicated=*/complicated, /*transposed=*/1);

  Matrix *base_relatives =
      ptr_to_matrices(input.model.base_relatives, input.model.n_bones, 4, 4);
  Matrix *inverse_base_absolutes = ptr_to_matrices(
      input.model.inverse_base_absolutes, input.model.n_bones, 4, 4);
  // Matrix base_positions =
  //     ptr_to_matrix(input.model.base_positions, 4, input.model.n_vertices);
  // Matrix weights = ptr_to_matrix(input.model.weights, input.model.n_bones,
  //                                input.model.n_vertices);
  // Matrix points = ptr_to_matrix(input.points, 3, input.n_pts);
  Triangle *triangles =
      ptr_to_triangles(input.model.triangles, input.model.n_triangles);
  // struct MatrixConverted converted = {.base_relatives = base_relatives,
  //                                     .inverse_base_absolutes =
  //                                         inverse_base_absolutes,
  //                                     .base_positions = &base_positions,
  //                                     .weights = &weights,
  //                                     .points = &points,
  //                                     .triangles = triangles};
  int J_rows = 3 * input.n_pts;
  int J_cols = complicated ? input.n_theta + 2 : input.n_theta;
  double *ref_J = (double *)malloc(J_rows * J_cols * sizeof(double));
  if (!CHECK_MEM) {
    populate_ref(&input, ref_J, complicated);
  }
  double *J = (double *)malloc(J_rows * J_cols * sizeof(double));

  unsigned long results_df[NUM_RUNS];
  bodyFunc funcs[] = {lagrad_hand_simple, enzyme_hand_simple};
  // bodyFunc funcs[] = {lagrad_hand_complicated, enzyme_mlir_hand_complicated};
  size_t num_apps = sizeof(funcs) / sizeof(funcs[0]);
  for (size_t app = 0; app < num_apps; app++) {
    for (size_t run = 0; run < NUM_RUNS; run++) {
      results_df[run] = (*funcs[app])(&input, J, ref_J);
    }
    print_ul_arr(results_df, NUM_RUNS);
  }

  // for (size_t run = 0; run < NUM_RUNS; run++) {
  //   if (complicated) {
  //     results_df[run] = enzyme_C_hand_complicated(&input, &converted, J, ref_J);
  //   } else {
  //     results_df[run] = enzyme_C_hand_simple(&input, &converted, J, ref_J);
  //   }
  // }
  // print_ul_arr(results_df, NUM_RUNS);

  /* Cleanup */
  free_matrix_array(base_relatives, input.model.n_bones);
  free_matrix_array(inverse_base_absolutes, input.model.n_bones);
  // free(base_positions.data);
  // free(weights.data);
  // free(points.data);
  free(triangles);
  free(J);
  free(ref_J);
  free_hand_input(&input);
  return 0;
}
