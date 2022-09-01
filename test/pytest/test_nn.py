import pytest
import os.path as osp
import torch
import numpy as np
from toolchain import jit_file
from mlir_bindings import mlir_mlp_primal, lagrad_mlp
from pytorch_ref.pytorch_nn import torch_mlp, torch_primal
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")
np.random.seed(0)


def test_cross_entropy():
    assert extract_1d(jit_file(f"{MLIR_FILES}/nn/crossentropy.mlir")) == pytest.approx(
        [0.00560261, -0.442628, 0.092133, 0.344892]
    )


def test_batched_cross_entropy():
    assert extract_2d(
        jit_file(f"{MLIR_FILES}/nn/batched_crossentropy.mlir")
    ) == pytest.approx(
        np.array(
            [
                [0.00280131, 0.360462],
                [-0.221314, 0.132607],
                [0.0460665, -0.499671],
                [0.172446, 0.0066021],
            ]
        )
    )


def test_mlp():
    parameters = [param.detach().numpy() for param in torch_mlp.parameters()]
    X = np.random.randn(784, 64).astype(np.float32)
    y = np.random.randint(low=0, high=10, size=(64,)).astype(np.int32)
    ref_primal = torch_primal(torch.from_numpy(X).mT, torch.from_numpy(y).long())
    mlir_params = [X, y] + parameters
    act_primal = mlir_mlp_primal(*mlir_params)
    assert act_primal == pytest.approx(ref_primal.item())
    dweight0, dbias0, dweight1, dbias1, dweight2, dbias2 = lagrad_mlp(*mlir_params)
    ref_primal.backward()
    ref_grads = [p.grad for p in torch_mlp.parameters()]

    # pytest.approx tests appear to be too slow for these bigger comparisons.
    tol = 3e-8
    assert np.abs(dweight0 - ref_grads[0].numpy()).max() < tol
    assert np.abs(dbias0 - ref_grads[1].numpy()).max() < tol
    assert np.abs(dweight1 - ref_grads[2].numpy()).max() < tol
    assert np.abs(dbias1 - ref_grads[3].numpy()).max() < tol
    assert np.abs(dweight2 - ref_grads[4].numpy()).max() < tol
    assert dbias2 == pytest.approx(ref_grads[-1], rel=1e-5)
