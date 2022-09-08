import pytest
import numpy as np
from mlir_bindings import rowhot_insert


def test_insert_slice():
    x = np.zeros((10, 3))
    x[4] = [5.6, 4.5, 2.3]
    expected = np.zeros((10, 4))
    expected[4, :-1] = x[4]
    assert rowhot_insert(x, np.int64(4)) == pytest.approx(expected)
