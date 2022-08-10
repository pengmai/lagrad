import pytest
import os.path as osp
import numpy as np
from toolchain import jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_cross_entropy():
    assert extract_1d(jit_file(f"{MLIR_FILES}/nn/crossentropy.mlir")) == pytest.approx(
        [0.00560261, -0.442628, 0.092133, 0.344892]
    )


def test_batched_cross_entropy():
    assert extract_2d(
        jit_file(f"{MLIR_FILES}/nn/batched_crossentropy.mlir")
    ) == pytest.approx(
        np.array(
            [
                [0.00280131, 0.360462],
                [-0.221314, 0.132607],
                [0.0460665, -0.499671],
                [0.172446, 0.0066021],
            ]
        )
    )
