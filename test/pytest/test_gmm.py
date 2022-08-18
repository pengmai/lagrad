import pytest
import os.path as osp
import pathlib
import torch
import numpy as np
from toolchain import jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d
from benchmark_io import read_gmm_instance, GMMInput
from pytorch_ref.pytorch_gmm import gmm_objective
from mlir_bindings import mlir_gmm_primal_full, lagrad_gmm_full

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")
GMM_DATA_FILE = (
    pathlib.Path(__file__).parents[2] / "benchmarks" / "data" / "gmm" / "test.txt"
)


@pytest.fixture(scope="module")
def gmm_input():
    return read_gmm_instance(GMM_DATA_FILE, False)


def test_main_term():
    stdout = jit_file(f"{MLIR_FILES}/gmm/main_term.mlir").split("Unranked")[1:]
    dalphas, dmeans, dQs, dLs = (
        extract_1d(stdout[0]),
        extract_2d(stdout[1]),
        extract_2d(stdout[2]),
        extract_3d(stdout[3]),
    )

    expected_alphas = [2.15774, 1.48867, 3.34759, 3.00601]
    expected_means = [
        [0.588905, 1.41125],
        [1.95973, -0.664794],
        [-3.87017, -3.95052],
        [1.2778, 0.208737],
    ]
    expected_Qs = [
        [1.41744, 1.68987],
        [0.855141, 0.827361],
        [2.641, 2.44121],
        [2.42314, 2.60779],
    ]
    expected_Ls = [
        [[-0.322486, -0.134414], [-0.0529821, -0.451955]],
        [[-0.600526, 0.0938678], [0.194995, -0.389366]],
        [[-0.361171, -0.256014], [-0.0637425, -0.899428]],
        [[-0.397244, 0.00946553], [-0.151172, -0.372482]],
    ]
    assert dalphas == expected_alphas
    assert dmeans == expected_means
    assert dQs == expected_Qs
    assert dLs == expected_Ls


def test_compressed_vecmat():
    expected = [-38.41, 29.46, -0.3, 5.4, 0]
    assert extract_1d(jit_file(f"{MLIR_FILES}/gmm/compressed_vecmat.mlir")) == expected


def test_compressed_outer_product():
    expected = [-10.23, 21.45, -13.86, 3.96, -20.8, 13.44, -3.84, -8.82, 2.52, 1.44]
    assert (
        extract_1d(jit_file(f"{MLIR_FILES}/gmm/compressed_outer_product.mlir"))
        == expected
    )


def materialize_L(k: int, d: int, L_compressed: np.ndarray):
    L_full = np.zeros((k, d, d))
    r, c = np.triu_indices(d, 1)
    L_full[:, r, c] = L_compressed
    return L_full.transpose(0, 2, 1)


def compress_L(L_full: np.ndarray):
    d = L_full.shape[1]
    assert L_full.shape[2] == d, "L_full shape was invalid"
    r, c = np.triu_indices(d, 1)
    L_compressed = L_full.transpose(0, 2, 1)[:, r, c]
    return L_compressed


def test_gmm_objective(gmm_input: GMMInput):
    # Arrange.
    k, d = gmm_input.means.shape
    L_full = materialize_L(k, d, gmm_input.icf[:, d:])
    args = (
        gmm_input.alphas,
        gmm_input.means,
        np.ascontiguousarray(gmm_input.icf[:, :d]),
        np.ascontiguousarray(L_full),
        gmm_input.x,
        np.float64(gmm_input.wishart.gamma),
        np.int64(gmm_input.wishart.m),
    )

    tactive = [
        torch.from_numpy(x)
        for x in [gmm_input.alphas, gmm_input.means, gmm_input.icf, gmm_input.x]
    ]
    for arg in tactive[:3]:
        arg.requires_grad = True
    torch_primal = gmm_objective(
        *(tactive + [gmm_input.wishart.gamma, gmm_input.wishart.m])
    )

    torch_primal.backward()

    # Act
    mlir_primal = mlir_gmm_primal_full(*args)
    dalphas, dmeans, dQs, dLs = lagrad_gmm_full(*args)

    # Assert
    assert mlir_primal == pytest.approx(torch_primal.item())
    assert dalphas == pytest.approx(tactive[0].grad)
    assert dmeans == pytest.approx(tactive[1].grad)
    assert dQs == pytest.approx(tactive[2].grad[:, :d])
    assert compress_L(dLs) == pytest.approx(tactive[2].grad[:, d:])
