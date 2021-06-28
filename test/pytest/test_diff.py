import os.path as osp
import re
import json
from toolchain import compile_pipeline

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def remove_whitespace(output: str):
    return re.sub(r"\s+", "", output)

def extract_scalar(output: str):
    # Remove all newlines
    output = remove_whitespace(output)
    float_regex = r"[-+]?[0-9]*\.?[0-9]+"
    pat = re.compile(rf"data=\[({float_regex})\]")
    m = pat.search(output)
    return float(m.group(1))


def extract_1d(output: str):
    output = remove_whitespace(output)
    float_regex = r"[-+]?[0-9]*\.?[0-9]+"
    pat = re.compile(rf"data=\[({float_regex}(?:,{float_regex})*)\]")
    m = pat.search(output).group(1)
    return [float(el) for el in m.split(",")]

def extract_2d(output: str):
    output = remove_whitespace(output)
    float_regex = r"[-+]?[0-9]*\.?[0-9]+"
    arr_regex = rf"\[{float_regex}(?:,{float_regex})*\]"
    mat_regex = re.compile(rf"data=(\[{arr_regex}(?:,{arr_regex})*\])")
    m = json.loads(mat_regex.search(output).group(1))
    return m

def test_square():
    output = compile_pipeline(f"{MLIR_FILES}/before.mlir")
    assert extract_scalar(output.decode("utf-8")) == 6.0


def test_twoargs():
    # TODO: No way to support derivatives w.r.t. multiple floats
    output = compile_pipeline(f"{MLIR_FILES}/twoargs.mlir")
    assert extract_scalar(output.decode("utf-8")) == 1.4


def test_1d_sum():
    output = compile_pipeline(f"{MLIR_FILES}/arrsum.mlir")
    assert extract_1d(output.decode('utf-8')) == [1.0] * 4


def test_scf_dot():
    output = compile_pipeline(f"{MLIR_FILES}/scfdot.mlir")
    assert extract_1d(output.decode("utf-8")) == [-0.3, 1.4, 2.2, -3.0]

def test_2d_sum():
    output = compile_pipeline(f"{MLIR_FILES}/matsum.mlir")
    assert extract_2d(output.decode("utf-8")) == [[1.0] * 4] * 4
