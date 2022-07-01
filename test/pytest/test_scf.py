import pytest
import os.path as osp
import numpy as np
from jinja2 import Template
from toolchain import jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_wrt_iter_arg():
    assert extract_2d(jit_file(f"{MLIR_FILES}/scf/wrt_iter_arg.mlir")) == [
        [4, 6, 9],
        [3, 6, 9],
    ]


def test_tensor_scf_for():
    assert (
        extract_2d(jit_file(f"{MLIR_FILES}/scf/fortensor.mlir"))
        == (np.ones((4, 4)) * 4).tolist()
    )


def test_for_intermediate():
    expected = [[5, 6, 7, 8], [1, 2, 3, 4]]
    assert extract_2d(jit_file(f"{MLIR_FILES}/scf/forintermediate.mlir")) == expected


def test_select_different_type():
    """
    This tests the case where the result of an scf.if op is a different type than
    the variable being differentiated.
    """
    assert (
        extract_scalar(jit_file(f"{MLIR_FILES}/scf/selectdifferenttype.mlir"))
        == 9.96035
    )


def test_nested_with_slice():
    expected = np.array(
        [
            [
                1.2094878928550226,
                -0.04627702933512943,
                -0.9271757135830436,
                -0.630883815588294,
            ],
            [
                -0.26723344833859586,
                1.1720411197584693,
                -0.30573774328968123,
                -0.05600016056043135,
            ],
            [
                0.03714989146333794,
                0.9853811447752215,
                0.6912993267213319,
                0.9305612728201504,
            ],
            [
                1.0199693238146688,
                0.1161007839852815,
                -0.9436884649582198,
                -0.6928985163024828,
            ],
            [
                -0.4793943443684846,
                1.2310141855084598,
                -0.052426193571350754,
                0.14762548022489902,
            ],
            [
                -0.10121638530602849,
                0.0906869077389665,
                -0.10486087228380073,
                -0.21796437610857303,
            ],
            [
                0.17330122739941822,
                -0.21166472544066658,
                0.6453014020338487,
                1.1002301319100263,
            ],
            [
                0.9667144381359907,
                0.19264673181472544,
                -0.7259967372786947,
                -0.4672098724831083,
            ],
            [
                0.4110007694100248,
                -0.752210110483746,
                0.5185245223382698,
                1.1159105995216938,
            ],
            [
                0.20960667677272032,
                0.7712357488338639,
                -0.33799537594391116,
                -0.09527366666465778,
            ],
        ]
    )
    assert np.array(
        extract_2d(jit_file(f"{MLIR_FILES}/scf/nested_with_slice.mlir"))
    ) == pytest.approx(expected, rel=1e-5)
