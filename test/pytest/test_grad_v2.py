"""
More recent tests for Grad, measuring the gradients of more advanced features such as
control flow and tensor.generate statements.

TODO: Should really consolidate the test_grad.py files, after fixing these tests.
Some files are missing from transferring computers.
"""

import os.path as osp
import numpy as np
from jinja2 import Template
from toolchain import jit_file, jit
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_generic_dot():
    assert extract_1d(jit_file(f"{MLIR_FILES}/generic/dot.mlir")) == [5, 6, 7, 8]


def test_generic_sub():
    assert extract_1d(jit_file(f"{MLIR_FILES}/generic/sub.mlir")) == [-1, -1, -1, -1]


def test_generic_vecmat():
    assert extract_1d(jit_file(f"{MLIR_FILES}/generic/vecmat.mlir")) == [10, 26, 42]


def test_generic_colsum():
    assert (
        extract_2d(jit_file(f"{MLIR_FILES}/generic/colsum.mlir"))
        == np.ones((4, 5)).tolist()
    )


def test_einsum_compare():
    """This is a test of a specific einsum found in the GMM benchmark."""
    n, k, d1, d2 = 2, 3, 4, 4
    a = np.arange(k * d1 * d2).reshape(k, d1, d2)
    b = np.arange(n * k * d2).reshape(n, k, d2)
    # print(a.tolist())
    # print(b.tolist())
    assert np.einsum("ijk,mik->mij", a, b).tolist() == extract_3d(
        jit_file(f"{MLIR_FILES}/generic/gmm_einsum.mlir")
    )


# def test_function_call():
#     print(jit_file(f"{MLIR_FILES}/functioncall.mlir"))


def test_broadcast_square():
    assert (
        extract_scalar(jit_file(f"{MLIR_FILES}/generic/broadcast_square.mlir")) == 920
    )


def test_three_args():
    with open(f"{MLIR_FILES}/generic/three_args.mlir") as f:
        template = Template(f.read())
    n, k, d = 4, 5, 6
    np.random.seed(0)
    x = np.random.rand(n, k, d)
    lx = np.random.rand(n, k, d)
    qs = np.random.rand(k, d)
    config = {
        "n": n,
        "k": k,
        "d": d,
        "Qdiags": qs.tolist(),
        "xcentered": x.tolist(),
        "Lxcentered_intermediate": lx.tolist(),
    }
    expected = np.array(
        [
            [2.41878, 2.85677, 3.28829, 2.14895, 1.17637, 3.32838],
            [4.05145, 1.80468, 5.16941, 0.990859, 4.65306, 2.9535],
            [5.45383, 6.55693, 4.47996, 2.51914, 1.49559, 2.71785],
            [3.55329, 3.4295, 3.53719, 1.918, 1.62634, 3.868],
            [2.28258, 2.38696, 2.69382, 4.36002, 2.7146, 4.85282],
        ]
    )

    mlir_res = np.array(extract_2d(jit(template.render(**config).encode("utf-8"))))
    assert np.abs(mlir_res - expected).sum() < 1e-7


def disabled_test_if_else():
    print("if-else", jit_file(f"{MLIR_FILES}/select.mlir"))
