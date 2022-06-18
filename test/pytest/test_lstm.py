import pytest
import warnings
from benchmark_io import read_lstm_instance, LSTMInput
from pytorch_ref.pytorch_lstm import lstm, predict
from mlir_bindings import lagrad_lstm_model, lagrad_lstm_predict
import torch
import os.path as osp

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")
LSTM_DATA_FILE = osp.join(
    osp.dirname(__file__), "..", "..", "benchmarks", "data", "lstm", "lstm_l2_c1024.txt"
)


@pytest.fixture(scope="module")
def lstm_input():
    return read_lstm_instance(LSTM_DATA_FILE)


def test_lstm_model(lstm_input: LSTMInput):
    # Arrange
    weight, bias, hidden, cell, _input = (
        lstm_input.main_params[0],
        lstm_input.main_params[1],
        lstm_input.state[0],
        lstm_input.state[1],
        lstm_input.sequence[0],
    )
    torch_args = []
    for arg in [weight, bias, hidden, cell, _input]:
        targ = torch.from_numpy(arg)
        targ.requires_grad = True
        torch_args.append(targ)
    lstm(*torch_args)[0].sum().backward()

    # Act
    dweight, dbias, dhidden, dcell, dinput = lagrad_lstm_model(
        weight.reshape(4, -1), bias.reshape(4, -1), hidden, cell, _input
    )

    # Assert
    tol = 1e-10
    assert dweight == pytest.approx(torch_args[0].grad.view(4, -1), tol)
    assert dbias == pytest.approx(torch_args[1].grad.view(4, -1), tol)
    assert dhidden == pytest.approx(torch_args[2].grad, tol)
    assert dcell == pytest.approx(torch_args[3].grad, tol)
    assert dinput == pytest.approx(torch_args[4].grad, tol)


def test_lstm_predict(lstm_input: LSTMInput):
    x = lstm_input.sequence[0]
    torch_args = []
    for i, arg in enumerate([lstm_input.main_params, lstm_input.extra_params, lstm_input.state, x]):
        targ = torch.from_numpy(arg)
        if i < 3:
            targ.requires_grad = True
        torch_args.append(targ)
    predict(*torch_args)[0].sum().backward()

    dmain, dextra, dstate = lagrad_lstm_predict(
        lstm_input.main_params.reshape(2, 2, 4, -1),
        lstm_input.extra_params,
        lstm_input.state.reshape(2, 2, -1).copy(),
        x,
    )

    tol = 1e-10
    assert dmain == pytest.approx(torch_args[0].grad.view(2, 2, 4, -1), tol)
    assert dextra == pytest.approx(torch_args[1].grad, tol)
    assert dstate == pytest.approx(torch_args[2].grad.view(2, 2, -1), tol)
