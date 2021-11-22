"""Tests designed to measure if activity analysis is correctly being performed."""
import os.path as osp
from toolchain import jit_file
from stdout_parser import extract_1d, extract_scalar
import warnings

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_unused_op():
    warnings.warn(UserWarning("Activity analysis test currently disabled"))
    # assert extract_1d(jit_file(f"{MLIR_FILES}/activity/unused.mlir")) == [
    #     -1.2,
    #     -1.2,
    #     -1.2,
    # ]


def test_ifelse():
    assert extract_scalar(jit_file(f"{MLIR_FILES}/activity/ifelse.mlir")) == 353.736
