import os.path as osp
import numpy as np
from toolchain import jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_to_pose_params():
    primal, adj = jit_file(f"{MLIR_FILES}/hand/to_pose_params.mlir").split("Unranked")[
        1:
    ]
    primal = extract_2d(primal)
    adj = extract_1d(adj)
    expected_primal = [
        [1, 2, 3],
        [1, 1, 1],
        [4, 5, 6],
        [0, 0, 0],
        [0, 0, 0],
        [7, 8, 0],
        [9, 0, 0],
        [10, 0, 0],
        [0, 0, 0],
        [11, 12, 0],
        [13, 0, 0],
        [14, 0, 0],
        [0, 0, 0],
        [15, 16, 0],
        [17, 0, 0],
        [18, 0, 0],
        [0, 0, 0],
        [19, 20, 0],
        [21, 0, 0],
        [22, 0, 0],
        [0, 0, 0],
        [23, 24, 0],
        [25, 0, 0],
        [26, 0, 0],
        [0, 0, 0],
    ]
    assert primal == expected_primal
    assert adj == np.ones(26).tolist()


def test_pose_relatives_body():
    expected = [40.8654, 29.4633, -25.8706]
    assert (
        extract_1d(jit_file(f"{MLIR_FILES}/hand/get_posed_relatives_body.mlir"))
        == expected
    )


def test_get_posed_relatives():
    expected = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [7.64197, -6.62924, -39.948],
        [10.2865, 50.6281, -136.472],
        [130.857, -149.433, -339.47],
        [-196.358, -92.8808, -411.033],
        [291.604, -413.609, -653.054],
        [-607.845, -505.337, -739.837],
        [578.474, -675.249, -845.945],
        [-930.038, -923.424, -895.76],
        [973.027, -850.391, -914.047],
        [-819.011, -980.066, -643.158],
        [1395.82, -873.335, -932.316],
        [-115.773, -462.371, 54.694],
        [1797.5, -652.014, -946.62],
        [1004.88, 500.791, 922.301],
        [2187.01, -91.9165, -926.029],
        [2098.76, 1466.5, 1467.93],
        [2553.83, 788.091, -829.505],
        [2710.84, 1919.31, 1258.32],
        [2755.56, 1735.47, -723.26],
        [2662.28, 1571.34, 200.056],
        [2504.22, 2277.98, -816.961],
        [2166.97, 520.21, -1370.05],
    ]
    assert extract_2d(jit_file(f"{MLIR_FILES}/hand/get_posed_relatives.mlir")) == expected


def test_angle_axis_to_rotation_matrix():
    expected_adj = [0.958705, 0.352867, -0.25297]
    expected_pri = [
        [-0.694921, -0.192007, 0.692978],
        [0.713521, -0.303785, 0.63135],
        [0.0892929, 0.933192, 0.348107],
    ]
    line1, line2 = jit_file(
        f"{MLIR_FILES}/hand/angle_axis_to_rotation_matrix.mlir"
    ).split("Unranked")[1:]
    primal = extract_2d(line1)
    adjoint = extract_1d(line2)
    assert primal == expected_pri
    assert adjoint == expected_adj


def test_apply_global_transform():
    expected_first = [
        [1.80638e06, 449664, -501519],
        [-86042.1, 462425, 609530],
        [544, 544, 544],
    ]
    adjoint = extract_2d(jit_file(f"{MLIR_FILES}/hand/apply_global_transform.mlir"))
    nonzero = adjoint[:3]
    zero_vals = adjoint[3:]
    assert nonzero == expected_first
    assert zero_vals == [[0] * len(zero_vals[0])] * len(zero_vals)


def test_compute_err_simple():
    adjoint = extract_2d(jit_file(f"{MLIR_FILES}/hand/compute_err_simple.mlir"))
    for i, row in enumerate(adjoint):
        if i in [10, 16]:
            assert row == [-1, -1, -1]
        else:
            assert row == [0, 0, 0]
