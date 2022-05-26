import os.path as osp
import numpy as np
from toolchain import jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_main_term():
    stdout = jit_file(f"{MLIR_FILES}/gmm/main_term.mlir").split("Unranked")[1:]
    dalphas, dmeans, dQs, dLs = (
        extract_1d(stdout[0]),
        extract_2d(stdout[1]),
        extract_2d(stdout[2]),
        extract_3d(stdout[3]),
    )

    expected_alphas = [2.15774, 1.48867, 3.34759, 3.00601]
    expected_means = [
        [0.588905, 1.41125],
        [1.95973, -0.664794],
        [-3.87017, -3.95052],
        [1.2778, 0.208737],
    ]
    expected_Qs = [
        [1.41744, 1.68987],
        [0.855141, 0.827361],
        [2.641, 2.44121],
        [2.42314, 2.60779],
    ]
    expected_Ls = [
        [[-0.322486, -0.134414], [-0.0529821, -0.451955]],
        [[-0.600526, 0.0938678], [0.194995, -0.389366]],
        [[-0.361171, -0.256014], [-0.0637425, -0.899428]],
        [[-0.397244, 0.00946553], [-0.151172, -0.372482]],
    ]
    assert dalphas == expected_alphas
    assert dmeans == expected_means
    assert dQs == expected_Qs
    assert dLs == expected_Ls


def test_compressed_vecmat():
    expected = [-38.41, 29.46, -0.3, 5.4, 0]
    assert extract_1d(jit_file(f"{MLIR_FILES}/gmm/compressed_vecmat.mlir")) == expected


def test_compressed_outer_product():
    expected = [-10.23, 21.45, -13.86, 3.96, -20.8, 13.44, -3.84, -8.82, 2.52, 1.44]
    assert (
        extract_1d(jit_file(f"{MLIR_FILES}/gmm/compressed_outer_product.mlir"))
        == expected
    )
