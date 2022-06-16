import pytest
import torch
from mlir_bindings import lagrad_ba_compute_reproj_error, lagrad_ba_compute_w_error
from benchmark_io import read_ba_instance, BAInput
from pytorch_ref.pytorch_ba import compute_reproj_err, compute_w_err
from pytorch_ref.utils import torch_jacobian
from pytorch_ref.BASparseMat import BASparseMat
import os.path as osp
import numpy as np

BA_DATA_FILE = osp.join(
    osp.dirname(__file__), "..", "..", "benchmarks", "data", "ba", "test.txt"
)


def compute_ba_jacobian(
    reproj_jacobian_row, weight_row, n, m, p, cams, x, w, obs, feats
):
    jacobian = BASparseMat(n, m, p)
    dreproj_error = np.empty((len(cams[0]) + len(x[0]) + 1, 2), dtype=np.float64)
    # reprojection error jacobian calculation
    for j in range(p):
        camIdx = obs[j, 0]
        ptIdx = obs[j, 1]
        reproj_jacobian_row(cams[camIdx], x[ptIdx], w[j], feats[j], dreproj_error)

        jacobian.insert_reproj_err_block(j, camIdx, ptIdx, dreproj_error.flatten())

    # weight error jacobian calculation
    for j in range(p):
        dw = weight_row(w[j])
        jacobian.insert_w_err_block(j, dw)

    return jacobian


@pytest.fixture(scope="module")
def ba_input():
    return read_ba_instance(BA_DATA_FILE)


def to_torch_tensor(el):
    t = torch.from_numpy(np.array(el))
    t.requires_grad = True
    return t


def reproj_pytorch_row(cam, xrow, wval, featrow, dreproj_err):
    _, J = torch_jacobian(compute_reproj_err, (cam, xrow, wval), (featrow,))
    dreproj_err[:] = J.view(-1, 2).numpy()


def w_pytorch_row(wval):
    _, J = torch_jacobian(compute_w_err, (wval,))
    return J.item()


def reproj_jacobian_row(cam, xrow, wval, featrow, dreproj_err):
    for row in range(2):
        g = np.zeros(2)
        g[row] = 1.0
        dcam, dx, dw = lagrad_ba_compute_reproj_error(cam, xrow, wval, featrow, g)
        dreproj_err[: cam.shape[0], row] = dcam
        dreproj_err[cam.shape[0] : cam.shape[0] + xrow.shape[0], row] = dx
        dreproj_err[cam.shape[0] + xrow.shape[0], row] = dw


def test_ba(ba_input: BAInput):
    # Arrange
    with open(BA_DATA_FILE) as f:
        n, m, p = [int(x) for x in f.readline().split()]

    targs = [reproj_pytorch_row, w_pytorch_row, n, m, p]

    for i, arg in enumerate(
        [ba_input.cams, ba_input.x, ba_input.w, ba_input.obs, ba_input.feats]
    ):
        if i < 3:
            targ = tuple(to_torch_tensor(el) for el in arg)
        else:
            targ = torch.from_numpy(arg)
        targs.append(targ)

    pytorch_jacobian = compute_ba_jacobian(*targs)

    # Act
    lagrad_jacobian = compute_ba_jacobian(
        reproj_jacobian_row,
        lagrad_ba_compute_w_error,
        n,
        m,
        p,
        ba_input.cams,
        ba_input.x,
        ba_input.w,
        ba_input.obs,
        ba_input.feats,
    )
    # Assert
    assert pytorch_jacobian.nrows == lagrad_jacobian.nrows
    assert pytorch_jacobian.ncols == lagrad_jacobian.ncols
    assert len(pytorch_jacobian.rows) == len(lagrad_jacobian.rows)
    assert len(pytorch_jacobian.cols) == len(lagrad_jacobian.cols)
    tol = 1e-16
    assert np.array(pytorch_jacobian.vals) == pytest.approx(np.array(lagrad_jacobian.vals), tol)
