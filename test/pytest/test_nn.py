import pytest
import os.path as osp
from toolchain import jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_cross_entropy():
    assert extract_1d(jit_file(f"{MLIR_FILES}/nn/crossentropy.mlir")) == pytest.approx(
        [0.00560261, -0.442628, 0.092133, 0.344892]
    )
