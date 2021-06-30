import os.path as osp
from toolchain import compile_pipeline
from stdout_parser import extract_scalar

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_add3():
    output = compile_pipeline(f"{MLIR_FILES}/add3.mlir", mode="grad")
    assert extract_scalar(output.decode("utf-8")) == 3.0


def test_quadratic():
    output = compile_pipeline(f"{MLIR_FILES}/quadratic.mlir", mode="grad")
    assert extract_scalar(output.decode("utf-8")) == 3.4
