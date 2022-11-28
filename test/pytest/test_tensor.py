"""
Tests that center around tensor slicing behaviour.
"""
import pytest
import os.path as osp
import numpy as np
from toolchain import jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "LAGrad")


def test_insert_overwrite_scalar():
    assert (
        extract_scalar(jit_file(f"{MLIR_FILES}/tensor/insert_overwrite_scalar.mlir"))
    ) == 2.0


def test_insert_overwrite_tensor():
    assert extract_2d(jit_file(f"{MLIR_FILES}/tensor/insert_overwrite.mlir")) == [
        [1, 1],
        [2, 2],
    ]


def test_insert_slice_loop():
    fst, snd = jit_file(f"{MLIR_FILES}/tensor/insert_slice_loop.mlir").split(
        "Unranked"
    )[1:]
    assert extract_1d(fst) == pytest.approx(np.ones(4) * 3)
    assert extract_2d(snd) == pytest.approx(np.zeros((3, 4)))
