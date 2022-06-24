from asyncio import base_subprocess
import pytest
import pathlib
import os.path as osp
import torch
import numpy as np
from benchmark_io import read_hand_instance, HandInput
from pytorch_ref.pytorch_hand import (
    to_pose_params,
    get_posed_relatives,
    relatives_to_absolutes,
    helper_get_transforms,
    skinned_vertex_subset,
    hand_objective,
)
from pytorch_ref.utils import torch_jacobian
from mlir_bindings import (
    hand_to_pose_params,
    lagrad_hand_to_pose_params,
    hand_get_posed_relatives,
    lagrad_get_posed_relatives,
    hand_relatives_to_absolutes,
    lagrad_relatives_to_absolutes,
    mlir_HELPER_get_transforms,
    lagrad_skinned_vertex_subset,
    mlir_hand_objective,
    lagrad_hand_objective,
)
from toolchain import jit_file
from stdout_parser import extract_1d, extract_2d

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


@pytest.fixture(scope="module")
def pose_params(hand_input):
    return hand_to_pose_params(hand_input.theta)


@pytest.fixture(scope="module")
def posed_relatives(hand_input, pose_params):
    base_relatives = np.ascontiguousarray(
        hand_input.data.model.base_relatives.transpose(0, 2, 1)
    )
    return hand_get_posed_relatives(base_relatives, pose_params)


@pytest.fixture(scope="module")
def transforms(hand_input):
    """Remember that this is transposed relative to the PyTorch version."""
    model = hand_input.data.model
    return mlir_HELPER_get_transforms(
        hand_input.theta,
        model.parents.astype(np.int32),
        # MLIR is implemented assuming these matrices are column-major.
        np.ascontiguousarray(model.base_relatives.transpose(0, 2, 1)),
        np.ascontiguousarray(model.inverse_base_absolutes.transpose(0, 2, 1)),
    )


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


def test_skinned_vertex_subset(hand_input: HandInput, transforms):
    model = hand_input.data.model
    ttransforms = helper_get_transforms(
        *(
            torch.from_numpy(arg)
            for arg in [
                hand_input.theta,
                model.base_relatives,
                model.parents,
                model.inverse_base_absolutes,
            ]
        )
    )
    ttransforms.requires_grad = True
    skinned_vertex_subset(
        *(
            [ttransforms]
            + [torch.from_numpy(arg) for arg in [model.base_positions, model.weights]]
        )
    ).sum().backward()

    lagrad_adj = lagrad_skinned_vertex_subset(
        transforms, model.base_positions, model.weights
    )
    assert lagrad_adj == pytest.approx(ttransforms.grad.transpose(1, 2))


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


def test_relatives_to_absolutes(hand_input: HandInput, posed_relatives):
    trels = torch.from_numpy(posed_relatives)
    trels.requires_grad = True
    tabs = relatives_to_absolutes(
        trels, torch.from_numpy(hand_input.data.model.parents)
    )
    tabs.sum().backward()

    mabs = hand_relatives_to_absolutes(
        posed_relatives, hand_input.data.model.parents.astype(np.int32)
    )
    lagrad_drels = lagrad_relatives_to_absolutes(
        posed_relatives, hand_input.data.model.parents.astype(np.int32)
    )
    assert mabs == pytest.approx(tabs.detach())
    assert lagrad_drels == pytest.approx(trels.grad)


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
