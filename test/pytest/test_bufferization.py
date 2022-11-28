"""
These tests are not directly related to LAGrad, but test aggressive bufferization
procedures designed to optimize its performance.
"""

import os.path as osp
from toolchain import jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d
import numpy as np

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "LAGrad")


def test_collapse_shape():
    expected = (
        np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0]) * 4
    ).tolist()
    assert (
        extract_1d(jit_file(f"{MLIR_FILES}/bufferize/collapse_shape.mlir")) == expected
    )


def test_grad_collapse_shape():
    expected = np.broadcast_to(np.array([0, 1, 0, 0]), (4, 4)).T.tolist()
    assert (
        extract_2d(jit_file(f"{MLIR_FILES}/bufferize/grad_collapse_shape.mlir"))
        == expected
    )


def test_collapse_shape_col():
    expected = [-1, -2, -3, -4]
    assert (
        extract_1d(jit_file(f"{MLIR_FILES}/bufferize/collapse_shape_col.mlir"))
        == expected
    )


def test_grad_collapse_shape_col():
    expected = np.broadcast_to(np.array([0, 0, 1, 0]), (4, 4)).tolist()
    assert (
        extract_2d(jit_file(f"{MLIR_FILES}/bufferize/grad_collapse_shape_col.mlir"))
        == expected
    )


def test_collapse_shape_2d():
    assert extract_2d(jit_file(f"{MLIR_FILES}/bufferize/collapse_shape_2d.mlir")) == [
        [2.5, 3],
        [-14, -16],
    ]


def test_grad_collapse_shape_2d():
    expected = np.zeros((4, 4, 4))
    expected[2] = 1
    assert (
        extract_3d(jit_file(f"{MLIR_FILES}/bufferize/grad_collapse_shape_2d.mlir"))
        == expected.tolist()
    )
