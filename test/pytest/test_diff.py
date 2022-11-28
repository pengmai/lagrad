import os.path as osp
import numpy as np
from toolchain import compile_pipeline
from stdout_parser import extract_scalar, extract_1d, extract_2d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "LAGrad")


def test_square():
    output = compile_pipeline(f"{MLIR_FILES}/before.mlir")
    assert extract_scalar(output.decode("utf-8")) == 6.0


def test_twofuncs():
    [line1, line2] = (
        compile_pipeline(f"{MLIR_FILES}/diff/two_funcs.mlir")
        .decode("utf-8")
        .split("Unranked")[1:]
    )
    assert extract_scalar(line1) == 6
    assert extract_scalar(line2) == 27


def test_twoargs():
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


def test_memref():
    output = compile_pipeline(f"{MLIR_FILES}/diff/memref.mlir")
    assert extract_scalar(output.decode("utf-8")) == 2.3


def test_linalg_dot():
    output = compile_pipeline(f"{MLIR_FILES}/diff/dot.mlir")
    assert extract_1d(output.decode("utf-8")) == [9.0, 10.0, -11.0, 12.0]


def test_generic():
    output = compile_pipeline(f"{MLIR_FILES}/diff/generic.mlir")
    assert (
        extract_2d(output.decode("utf-8"))
        == np.broadcast_to([10, 4, 20, 58], (4, 4)).tolist()
    )


def test_matvec():
    output = compile_pipeline(f"{MLIR_FILES}/diff/matvec_buffer.mlir")
    assert (
        extract_2d(output.decode("utf-8"))
        == np.broadcast_to([-1.2, -1.3, 1.5, 2.2], (3, 4)).tolist()
    )


def test_const_memref():
    output = compile_pipeline(f"{MLIR_FILES}/diff/const_memref.mlir")
    assert extract_1d(output.decode("utf-8")) == [4.3, 4.3]


def test_softmax():
    output = compile_pipeline(f"{MLIR_FILES}/diff/softmax.mlir")
    print(output)
    # print(extract_1d(output.decode("utf-8")))
