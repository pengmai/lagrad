"""
More recent tests for Grad, measuring the gradients of more advanced features such as
control flow and tensor.generate statements.

TODO: Should really consolidate the test_grad.py files, after fixing these tests.
Some files are missing from transferring computers.
"""

import os.path as osp
import numpy as np
from toolchain import jit_file
from stdout_parser import extract_1d, extract_2d, extract_3d

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


def test_function_call():
    print(jit_file(f"{MLIR_FILES}/functioncall.mlir"))


def disabled_test_if_else():
    print(jit_file(f"{MLIR_FILES}/select.mlir"))
