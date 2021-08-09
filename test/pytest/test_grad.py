import numpy as np
import os.path as osp
from toolchain import compile_pipeline
from stdout_parser import extract_scalar, extract_1d, extract_2d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_add3():
    output = compile_pipeline(f"{MLIR_FILES}/add3.mlir", mode="grad")
    assert extract_scalar(output.decode("utf-8")) == 3.0


def test_quadratic():
    output = compile_pipeline(f"{MLIR_FILES}/quadratic.mlir", mode="grad")
    assert extract_scalar(output.decode("utf-8")) == 3.4


def test_logistic():
    output = compile_pipeline(f"{MLIR_FILES}/logistic.mlir", mode="grad")
    print(output.decode("utf-8"))


def test_dot():
    output = compile_pipeline(f"{MLIR_FILES}/dot.mlir", mode="grad")
    assert extract_1d(output.decode("utf-8")) == [-5.0, 3.4, -10.2, 3.33]


def test_dot_second_arg():
    output = compile_pipeline(f"{MLIR_FILES}/dot_second_arg.mlir", mode="grad")
    assert extract_1d(output.decode("utf-8")) == [0.1, 1.0, 2.0, -3.0]


def test_matvec_with_postmul():
    output = compile_pipeline(f"{MLIR_FILES}/matvec.mlir", mode="grad")
    assert np.allclose(
        extract_2d(output.decode("utf-8")),
        np.outer([1.1, -1.2, 1.0], [-1.2, -1.3, 1.5, 2.2]),
    )
