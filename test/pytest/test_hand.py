import pytest
import pathlib
import os.path as osp
import torch
import numpy as np
from benchmark_io import read_hand_instance, HandInput
from pytorch_ref.pytorch_hand import to_pose_params, get_posed_relatives, hand_objective
from pytorch_ref.utils import torch_jacobian
from mlir_bindings import (
    hand_to_pose_params,
    lagrad_hand_to_pose_params,
    hand_get_posed_relatives,
    lagrad_get_posed_relatives,
    mlir_hand_objective,
    lagrad_hand_objective,
)
from toolchain import jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d

HAND_DATA_DIR = (
    pathlib.Path(__file__).parents[2] / "benchmarks" / "data" / "hand" / "simple_small"
)
MODEL_DIR = HAND_DATA_DIR / "model"
HAND_DATA_FILE = HAND_DATA_DIR / "test.txt"
MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


@pytest.fixture(scope="module")
def hand_input():
    inp = read_hand_instance(MODEL_DIR, HAND_DATA_FILE, read_us=False)
    return inp


@pytest.fixture
def pose_params(hand_input):
    return hand_to_pose_params(hand_input.theta)


def test_to_pose_params(hand_input: HandInput):
    ttheta = torch.from_numpy(hand_input.theta)
    ttheta.requires_grad = True
    torch_primal = to_pose_params(ttheta, hand_input.data.model.bone_count)
    torch_primal.sum().backward()

    mlir_primal = hand_to_pose_params(hand_input.theta)
    mlir_grad = lagrad_hand_to_pose_params(hand_input.theta)
    tol = 1e-10
    assert mlir_primal == pytest.approx(torch_primal.detach(), tol)
    assert mlir_grad == pytest.approx(ttheta.grad, tol)


def test_pose_relatives_body():
    expected = [40.8654, 29.4633, -25.8706]
    assert (
        extract_1d(jit_file(f"{MLIR_FILES}/hand/get_posed_relatives_body.mlir"))
        == expected
    )


def test_get_posed_relatives(hand_input: HandInput, pose_params):
    tpp = torch.from_numpy(pose_params)
    tpp.requires_grad = True
    torch_posed_relatives = get_posed_relatives(
        tpp, torch.from_numpy(hand_input.data.model.base_relatives)
    )
    torch_posed_relatives.sum().backward()
    base_relatives = np.ascontiguousarray(
        hand_input.data.model.base_relatives.transpose(0, 2, 1)
    )
    mlir_posed_relatives = hand_get_posed_relatives(base_relatives, pose_params)

    dpose_params = lagrad_get_posed_relatives(
        base_relatives,
        pose_params,
    )
    assert mlir_posed_relatives == pytest.approx(
        torch_posed_relatives.transpose(1, 2).detach()
    )
    assert dpose_params == pytest.approx(tpp.grad, rel=1e-5)


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


def test_relatives_to_absolutes():
    expected = [
        [
            [11.8063, -3.14422, -14.0792, 22],
            [11.8063, -3.14422, -14.0792, 22],
            [11.8063, -3.14422, -14.0792, 22],
            [11.8063, -3.14422, -14.0792, 22],
        ],
        [
            [4.3483, -0.191615, 5.99559, 4.73371],
            [-3.76106, 0.165737, -5.18589, -4.09442],
            [-2.72924, 0.120268, -3.76318, -2.97115],
            [3.67433, -0.161915, 5.0663, 4],
        ],
        [
            [1.48676, 1.98744, 1.71621, 1.79999],
            [4.02249, 5.3771, 4.64328, 4.86995],
            [0.172607, 0.230734, 0.199245, 0.208972],
            [2.46875, 3.30012, 2.84975, 2.98886],
        ],
        [
            [-3.10782, -1.36636, -3.56762, -3.01867],
            [0.951197, 0.418195, 1.09193, 0.923911],
            [1.4683, 0.645542, 1.68554, 1.42618],
            [2.16936, 0.953762, 2.49032, 2.10713],
        ],
        [
            [-1.29444, -1.29444, -1.29444, -1.29444],
            [0.0913131, 0.0913131, 0.0913131, 0.0913131],
            [1.1472, 1.1472, 1.1472, 1.1472],
            [1.07115, 1.07115, 1.07115, 1.07115],
        ],
        [
            [3.9169, 3.84366, 5.41512, 4.73371],
            [-3.38792, -3.32457, -4.68381, -4.09442],
            [-2.45847, -2.4125, -3.39884, -2.97115],
            [3.30979, 3.2479, 4.5758, 4],
        ],
        [
            [2.06587, 1.0907, 2.18048, 2.05274],
            [3.7395, 1.97432, 3.94696, 3.71573],
            [3.01577, 1.59222, 3.18308, 2.9966],
            [3.00798, 1.5881, 3.17486, 2.98886],
        ],
        [
            [2.54374, 0.448973, 2.88817, 2.56346],
            [1.88637, 0.332947, 2.14179, 1.90099],
            [1.33682, 0.235951, 1.51783, 1.34719],
            [2.24515, 0.396272, 2.54915, 2.26256],
        ],
        [
            [1.23708, 1.23708, 1.23708, 1.23708],
            [0.91654, 0.91654, 0.91654, 0.91654],
            [0.793463, 0.793463, 0.793463, 0.793463],
            [1.16202, 1.16202, 1.16202, 1.16202],
        ],
        [
            [5.82161, 1.70028, 4.32822, 4.73371],
            [-5.0354, -1.47066, -3.74369, -4.09442],
            [-3.65398, -1.0672, -2.71664, -2.97115],
            [4.91928, 1.43675, 3.65736, 4],
        ],
        [
            [2.77198, 2.41893, 2.3971, 2.78113],
            [3.19407, 2.78727, 2.76211, 3.20461],
            [2.98943, 2.60869, 2.58515, 2.9993],
            [2.97902, 2.59961, 2.57615, 2.98886],
        ],
        [
            [0.0869321, 0.100043, 0.00898063, 0.0893064],
            [2.74241, 3.15601, 0.283309, 2.81732],
            [1.96011, 2.25572, 0.202491, 2.01364],
            [2.16289, 2.48909, 0.22344, 2.22196],
        ],
        [
            [0.0053296, 0.0053296, 0.0053296, 0.0053296],
            [1.72765, 1.72765, 1.72765, 1.72765],
            [-0.123242, -0.123242, -0.123242, -0.123242],
            [1.15903, 1.15903, 1.15903, 1.15903],
        ],
        [
            [4.56079, 5.14674, -0.897398, 4.73371],
            [-3.94486, -4.45167, 0.776204, -4.09442],
            [-2.86262, -3.23039, 0.563258, -2.97115],
            [3.85389, 4.34901, -0.758304, 4],
        ],
        [
            [3.36985, 3.975, 0.024609, 3.46339],
            [2.36746, 2.79261, 0.0172889, 2.43318],
            [2.93265, 3.45929, 0.0214163, 3.01406],
            [2.90813, 3.43037, 0.0212372, 2.98886],
        ],
        [
            [2.3511, 2.83955, 0.856018, 2.35435],
            [0.299979, 0.362301, 0.10922, 0.300394],
            [2.51977, 3.04326, 0.917429, 2.52325],
            [2.16574, 2.61568, 0.788528, 2.16873],
        ],
        [
            [1.17906, 1.17906, 1.17906, 1.17906],
            [-0.718315, -0.718315, -0.718315, -0.718315],
            [1.04587, 1.04587, 1.04587, 1.04587],
            [1.08861, 1.08861, 1.08861, 1.08861],
        ],
        [
            [-3.79135, 0.0872261, 3.99853, 4.73371],
            [3.27932, -0.0754461, -3.45853, -4.09442],
            [2.37967, -0.0547481, -2.50971, -2.97115],
            [-3.2037, 0.0737063, 3.37877, 4],
        ],
        [
            [4.03301, 4.66859, 1.29538, 3.79235],
            [2.05855, 2.38297, 0.661197, 1.93571],
            [3.16744, 3.66662, 1.01737, 2.97844],
            [3.17853, 3.67945, 1.02093, 2.98886],
        ],
        [
            [-1.81503, -1.43897, -2.02766, -1.79095],
            [-1.24985, -0.990892, -1.39627, -1.23327],
            [2.73284, 2.16662, 3.05298, 2.69658],
            [2.15626, 1.70951, 2.40886, 2.12765],
        ],
        [
            [-0.862798, -0.862798, -0.862798, -0.862798],
            [-1.48086, -1.48086, -1.48086, -1.48086],
            [0.250251, 0.250251, 0.250251, 0.250251],
            [1.05034, 1.05034, 1.05034, 1.05034],
        ],
        [
            [1.18343, 1.18343, 1.18343, 1.18343],
            [-1.02361, -1.02361, -1.02361, -1.02361],
            [-0.742787, -0.742787, -0.742787, -0.742787],
            [1, 1, 1, 1],
        ],
    ]
    adjoint = extract_3d(jit_file(f"{MLIR_FILES}/hand/relatives_to_absolutes.mlir"))
    assert adjoint == expected


def test_hand_objective_simple(hand_input: HandInput):
    ttheta = torch.from_numpy(hand_input.theta)
    ttheta.requires_grad = True
    model = hand_input.data.model
    tparams = [
        (torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg)
        for arg in [
            model.bone_count,
            model.parents,
            model.base_relatives,
            model.inverse_base_absolutes,
            model.base_positions,
            model.weights,
            model.is_mirrored,
            hand_input.data.points,
            hand_input.data.correspondences,
        ]
    ]

    torch_primal, torch_J = torch_jacobian(hand_objective, (ttheta,), tparams, False)
    mparams = (
        hand_input.theta,
        model.parents.astype(np.int32),
        # MLIR is implemented assuming these matrices are column-major.
        np.ascontiguousarray(model.base_relatives.transpose(0, 2, 1)),
        np.ascontiguousarray(model.inverse_base_absolutes.transpose(0, 2, 1)),
        model.base_positions,
        model.weights,
        hand_input.data.correspondences.astype(np.int32),
        hand_input.data.points,
    )

    mprimal = mlir_hand_objective(*mparams)
    err_size = np.prod(hand_input.data.points.shape)
    lagrad_J = np.empty((err_size, hand_input.theta.shape[0]))
    for i in range(err_size):
        g = np.zeros_like(hand_input.data.points)
        stride = hand_input.data.points.shape[1]
        g[i // stride, i % stride] = 1.0
        lagrad_J[i, :] = lagrad_hand_objective(*(mparams + (g,)))
    tol = 1e-8
    assert mprimal == pytest.approx(torch_primal.detach(), tol)
    assert lagrad_J == pytest.approx(torch_J)
