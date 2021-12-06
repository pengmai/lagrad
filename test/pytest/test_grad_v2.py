"""
More recent tests for Grad, measuring the gradients of more advanced features such as
control flow and tensor.generate statements.

TODO: Should really consolidate the test_grad.py files, after fixing these tests.
Some files are missing from transferring computers.
"""

import os.path as osp
import numpy as np
from jinja2 import Template
from toolchain import jit_file, jit
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_generic_dot():
    assert extract_1d(jit_file(f"{MLIR_FILES}/generic/dot.mlir")) == [5, 6, 7, 8]


def test_generic_sub():
    assert extract_1d(jit_file(f"{MLIR_FILES}/generic/sub.mlir")) == [-1, -1, -1, -1]


def test_generic_vecmat():
    assert extract_1d(jit_file(f"{MLIR_FILES}/generic/vecmat.mlir")) == [10, 26, 42]


def test_generic_colsum():
    assert (
        extract_2d(jit_file(f"{MLIR_FILES}/generic/colsum.mlir"))
        == np.ones((4, 5)).tolist()
    )


def test_einsum_compare():
    """This is a test of a specific einsum found in the GMM benchmark."""
    n, k, d1, d2 = 2, 3, 4, 4
    a = np.arange(k * d1 * d2).reshape(k, d1, d2)
    b = np.arange(n * k * d2).reshape(n, k, d2)
    # print(a.tolist())
    # print(b.tolist())
    assert np.einsum("ijk,mik->mij", a, b).tolist() == extract_3d(
        jit_file(f"{MLIR_FILES}/generic/gmm_einsum.mlir")
    )


def test_generic_function_call():
    expected = np.array(
        [
            [3.30259, 3.30259, 3.30259, 3.30259],
            [4.2581, 4.2581, 4.2581, 4.2581],
            [4.73767, 4.73767, 4.73767, 4.73767],
            [5.06044, 5.06044, 5.06044, 5.06044],
        ]
    )
    assert (
        np.abs(
            np.array(extract_2d(jit_file(f"{MLIR_FILES}/generic/functioncall.mlir")))
            - expected
        ).mean()
        < 1e-9
    )


def test_function_call():
    assert extract_scalar(jit_file(f"{MLIR_FILES}/functioncall.mlir")) == 1192.58


def test_nested_call():
    assert extract_scalar(jit_file(f"{MLIR_FILES}/nestedcall.mlir")) == 2.4


def test_if_function_call():
    assert extract_scalar(jit_file(f"{MLIR_FILES}/if_function.mlir")) == 8.4


def test_broadcast():
    assert extract_scalar(jit_file(f"{MLIR_FILES}/generic/broadcast.mlir")) == 4


def test_broadcast_square():
    assert (
        extract_scalar(jit_file(f"{MLIR_FILES}/generic/broadcast_square.mlir")) == 920
    )


def test_three_args():
    with open(f"{MLIR_FILES}/generic/three_args.mlir") as f:
        template = Template(f.read())
    n, k, d = 4, 5, 6
    np.random.seed(0)
    x = np.random.rand(n, k, d)
    qs = np.random.rand(k, d)
    config = {
        "n": n,
        "k": k,
        "d": d,
        "Qdiags": qs.tolist(),
        "xcentered": x.tolist(),
    }
    cst = np.ones((k, d, d)) * 2.3

    def hand_grad():
        g = np.ones((n, k, d))
        dx_qs = qs * g
        dx_einsum = np.einsum("ijk,mik->mij", cst, g)
        return dx_qs + dx_einsum

    hand = hand_grad()
    mlir_res = np.array(extract_3d(jit(template.render(**config).encode("utf-8"))))
    assert np.abs(mlir_res - hand).mean() < 1e-4


def test_logsumexp():
    with open(f"{MLIR_FILES}/generic/logsumexp.mlir") as f:
        template = Template(f.read())
    np.random.seed(0)
    x = np.random.rand(3, 4)
    expected = np.array(
        [
            [0.236266, 0.279034, 0.249362, 0.235339],
            [0.205766, 0.256975, 0.208653, 0.328606],
            [0.327953, 0.18358, 0.276146, 0.212321],
        ]
    )

    mlir_res = np.array(
        extract_2d(jit(template.render(data=x.tolist()).encode("utf-8")))
    )

    assert np.abs(expected - mlir_res).sum() < 1e-7


def disabled_test_nonconst_out():
    from autograd import elementwise_grad
    import autograd.numpy as np

    with open(f"{MLIR_FILES}/generic/nonconst_out.mlir") as f:
        template = Template(f.read())
    np.random.seed(0)
    A = np.array(
        [
            [0.377, 0.283, 0.155, 0.858],
            [0.3, 0.431, 0.851, 0.137],
            [0.776, 0.555, 0.771, 0.233],
            [0.623, 0.193, 0.005, 0.691],
        ]
    )
    B = np.array([4.0, -3.0, 2.0, 1.0])
    C = np.array([1.0, 0.2, -2.3, -1.7])

    print(elementwise_grad(lambda x, y, z: y * z + np.dot(x, y), 1)(A, B, C))
    print(np.array([2.076, 1.462, 1.782, 1.919]) + C)


def test_logsumexp_1d():
    expected = np.array([9.54081e-01, 1.30126e-04, 1.05989e-02, 3.51895e-02])
    assert (
        np.abs(
            np.array(extract_1d(jit_file(f"{MLIR_FILES}/generic/logsumexp_1d.mlir")))
            - expected
        ).sum()
        < 1e-6
    )


def test_cross():
    assert extract_1d(jit_file(f"{MLIR_FILES}/cross_product.mlir")) == [-1, 2, -1]


def test_batch_matmul():
    A = np.random.rand(4, 3, 5)
    B = np.reshape(np.arange(60).astype(np.float64) + 1, (4, 5, 3))
    B = np.where(B % 7 == 0, -B, B)
    g = np.ones((3, 3))
    expected = np.stack([g @ B[i].T for i in range(len(A))]).tolist()
    assert extract_3d(jit_file(f"{MLIR_FILES}/batch_matmul.mlir")) == expected


def test_extract_scalar():
    [line1, line2] = jit_file(f"{MLIR_FILES}/tensor_extract.mlir").split("Unranked")[1:]
    assert extract_scalar(line1) == 15.64
    assert extract_1d(line2) == [0, 0, 0, 7.48]


def test_extract_slice():
    assert extract_2d(jit_file(f"{MLIR_FILES}/tensor_extract_slice.mlir")) == [
        [1, 2],
        [1, 2],
    ]


def disabled_test_scalar_recursion():
    print(jit_file(f"{MLIR_FILES}/recursion.mlir"))


def test_relu():
    assert extract_1d(jit_file(f"{MLIR_FILES}/relu.mlir")) == [1, 0, 1, 0]


def test_closure():
    assert extract_scalar(jit_file(f"{MLIR_FILES}/generic/closure.mlir")) == 40


def test_custom_grad_init():
    assert extract_1d(jit_file(f"{MLIR_FILES}/custom_grad_init.mlir")) == [2.3, 0.0]


def test_insert_multi():
    assert extract_1d(jit_file(f"{MLIR_FILES}/insert_multi.mlir")) == [1, 0, 1]


def test_insert_slice():
    assert extract_1d(jit_file(f"{MLIR_FILES}/tensor/insert_slice.mlir")) == [
        1,
        1,
        1,
        1,
    ]


def test_if_else():
    output = jit_file(f"{MLIR_FILES}/ifelse.mlir").split("Unranked")[1:]
    parsed = [extract_scalar(line) for line in output]
    assert parsed == [9.0, 1.0, 0.0, 15.3, 353.736]


def test_if_different_type():
    """
    This tests the case where the result of an scf.if op is a different type than
    the variable being differentiated.
    """
    assert extract_scalar(jit_file(f"{MLIR_FILES}/ifdifferenttype.mlir")) == 9.96035


def test_tensor_scf_for():
    assert (
        extract_2d(jit_file(f"{MLIR_FILES}/scf/fortensor.mlir"))
        == (np.ones((4, 4)) * 4).tolist()
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
    assert (
        np.abs(
            np.array(extract_2d(jit_file(f"{MLIR_FILES}/scf/nested_with_slice.mlir")))
            - expected
        ).sum()
        < 1e-4
    )


def disabled_test_if_else():
    print("if-else", jit_file(f"{MLIR_FILES}/select.mlir"))
