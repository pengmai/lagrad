import pytest
import numpy as np
from mlir_bindings import (
    rowhot_insert,
    onehot_square,
    onehot_matmul,
    onehot_matmul_both_transposed,
    rowhot_broadcast_mul,
    rowhot_matmul,
)


def test_insert_slice():
    x = np.zeros((10, 3))
    x[4] = [5.6, 4.5, 2.3]
    expected = np.zeros((10, 4))
    expected[4, :-1] = x[4]
    assert rowhot_insert(x, np.int64(4)) == pytest.approx(expected)


def test_onehot_square():
    x = np.zeros((10, 3))
    x[4, 2] = 5.6
    assert onehot_square(x, np.array([4, 2]).astype(np.int64)) == pytest.approx(x ** 2)


def test_onehot_matmul():
    np.random.seed(0)
    y = np.random.rand(544, 3)
    x = np.zeros((544, 3))
    indices = np.array([2, 0]).astype(np.int64)
    x[indices[0], indices[1]] = -5.4
    assert onehot_matmul(y, x, indices) == pytest.approx(np.matmul(y.T, x))


def test_onehot_matmul_transpose():
    y = np.array(
        [[0.447, 0.688, -0.003], [0.63, -1.242, 2.959], [-1.285, -0.789, 1.211]]
    )
    x = np.zeros((544, 3))
    indices = np.array([68, 1]).astype(np.int64)
    x[indices[0], indices[1]] = -4.3
    assert onehot_matmul_both_transposed(y, x, indices) == pytest.approx(
        np.matmul(y, x.T).T
    )


def test_rowhot_broadcast_mul():
    np.random.seed(0)
    y = np.random.randn(12)
    x = np.zeros((12, 3))
    idx = np.int64(3)
    x[idx] = np.random.randn(3)
    assert rowhot_broadcast_mul(y, x, idx) == pytest.approx((y * x.T).T)


def test_rowhot_matmul():
    np.random.seed(0)
    y = np.random.randn(544, 4)
    x = np.zeros((544, 4))
    idx = np.int64(2)
    x[idx] = np.random.randn(4)
    assert rowhot_matmul(y, x, idx) == pytest.approx(np.matmul(y.T, x))
