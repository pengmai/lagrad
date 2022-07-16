"""
More recent tests for Grad, measuring the gradients of more advanced features such as
control flow and tensor.generate statements.

TODO: Should really consolidate the test_grad.py files, after fixing these tests.
Some files are missing from transferring computers.
"""

import pytest
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


def test_nonconst_out():
    expected = np.array([3.076, 1.662, -0.518, 0.219])
    assert np.array(
        extract_1d(jit_file(f"{MLIR_FILES}/generic/nonconst_out.mlir"))
    ) == pytest.approx(expected)


def test_logsumexp_add2():
    expected = np.array([0.0313198, 0.0851362, 0.231424, 0.629076])
    assert np.array(
        extract_1d(jit_file(f"{MLIR_FILES}/generic/logsumexp_add2.mlir"))
    ) == pytest.approx(expected)


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


def disabled_test_scalar_scf_for():
    expected = 12 * 1.1 ** 11

    def test(x):
        r = 1
        stack = []
        for _ in range(12):
            stack.append(r)
            r *= x

        dx = 0
        dr = 1
        for _ in range(12):
            r = stack.pop()
            dx += dr * r
            print(dr)
            dr *= x
        return dx

    print(test(1.1), (test(1.1) - expected) < 1e-6)
    # assert extract_scalar(jit_file(f"{MLIR_FILES}/scf/forscalar.mlir")) == expected


def test_pow_tensor():
    expected = 4 * np.array([1.3, 1.4, 1.5, 1.6]) ** 3
    assert np.array(
        extract_1d(jit_file(f"{MLIR_FILES}/cache/pow_tensor.mlir"))
    ) == pytest.approx(expected)


def test_nested_pow_scalar():
    expected = 12 * 1.3 ** 11
    assert extract_scalar(
        jit_file(f"{MLIR_FILES}/cache/nested_pow.mlir")
    ) == pytest.approx(expected, abs=1e-3)


def test_nested_pow_tensor():
    expected = np.ones(4) * 12 * 1.3 ** 11
    assert np.array(
        extract_1d(jit_file(f"{MLIR_FILES}/cache/nested_pow_tensor.mlir"))
    ) == pytest.approx(expected, abs=1e-3)


def disabled_test_if_else():
    print("if-else", jit_file(f"{MLIR_FILES}/select.mlir"))
