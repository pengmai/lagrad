import numpy as np
import os.path as osp
from toolchain import compile_pipeline, jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_add3():
    assert extract_scalar(jit_file(f"{MLIR_FILES}/add3.mlir")) == 3.0


def test_quadratic():
    assert extract_scalar(jit_file(f"{MLIR_FILES}/quadratic.mlir")) == 3.4


def disabled_test_logistic():
    print(jit_file(f"{MLIR_FILES}/logistic.mlir"))


def test_dot():
    assert extract_1d(jit_file(f"{MLIR_FILES}/dot.mlir")) == [-5.0, 3.4, -10.2, 3.33]


def test_dot_second_arg():
    assert extract_1d(jit_file(f"{MLIR_FILES}/dot_second_arg.mlir")) == [
        0.1,
        1.0,
        2.0,
        -3.0,
    ]


def test_matvec_with_postmul():
    assert np.allclose(
        extract_2d(jit_file(f"{MLIR_FILES}/matvec.mlir")),
        np.outer([1.1, -1.2, 1.0], [-1.2, -1.3, 1.5, 2.2]),
    )
