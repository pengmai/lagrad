"""
These tests are not directly related to LAGrad, but test aggressive bufferization
procedures designed to optimize its performance.
"""

import os.path as osp
from toolchain import jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d
import numpy as np

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


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
