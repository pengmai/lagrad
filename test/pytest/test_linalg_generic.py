import numpy as np
import os.path as osp
from toolchain import jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_dot():
    assert extract_1d(jit_file(f"{MLIR_FILES}/dot.mlir")) == [-5.0, 3.4, -10.2, 3.33]


def test_vecmat():
    assert extract_1d(jit_file(f"{MLIR_FILES}/vecmat.mlir")) == [19.2, -2.5, -4.5, 5.5]


def test_matvec_with_postmul():
    assert np.allclose(
        extract_2d(jit_file(f"{MLIR_FILES}/matvec.mlir")),
        np.outer([1.1, -1.2, 1.0], [-1.2, -1.3, 1.5, 2.2]),
    )


def test_matmul():
    assert extract_2d(jit_file(f"{MLIR_FILES}/matmul.mlir")) == [
        [1.69783, 3.7757, 3.0442, 2.06436],
        [1.69783, 3.7757, 3.0442, 2.06436],
        [1.69783, 3.7757, 3.0442, 2.06436],
    ]
