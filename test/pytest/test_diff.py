import os.path as osp
import numpy as np
from toolchain import compile_pipeline
from stdout_parser import extract_scalar, extract_1d, extract_2d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_square():
    output = compile_pipeline(f"{MLIR_FILES}/before.mlir")
    assert extract_scalar(output.decode("utf-8")) == 6.0


def test_twoargs():
    # TODO: No way to support derivatives w.r.t. multiple floats
    output = compile_pipeline(f"{MLIR_FILES}/twoargs.mlir")
    assert extract_scalar(output.decode("utf-8")) == 1.4


def test_1d_sum():
    output = compile_pipeline(f"{MLIR_FILES}/arrsum.mlir")
    assert extract_1d(output.decode("utf-8")) == [1.0] * 4


def test_scf_dot():
    output = compile_pipeline(f"{MLIR_FILES}/scfdot.mlir")
    assert extract_1d(output.decode("utf-8")) == [-0.3, 1.4, 2.2, -3.0]


def test_2d_sum():
    output = compile_pipeline(f"{MLIR_FILES}/matsum.mlir")
    assert extract_2d(output.decode("utf-8")) == [[1.0] * 4] * 4


def disabled_test_memref():
    output = compile_pipeline(f"{MLIR_FILES}/diff/memref.mlir")
    print(output.decode("utf-8"))


def test_linalg_dot():
    output = compile_pipeline(f"{MLIR_FILES}/diff/dot.mlir")
    assert extract_1d(output.decode("utf-8")) == [9.0, 10.0, -11.0, 12.0]


def test_generic():
    output = compile_pipeline(f"{MLIR_FILES}/diff/generic.mlir")
    extract_2d(output.decode("utf-8")) == np.broadcast_to([10, 4, 20, 58], (4, 4))
