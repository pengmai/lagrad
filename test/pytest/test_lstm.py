import pytest
from benchmark_io import read_lstm_instance, LSTMInput
from pytorch_ref.pytorch_lstm import lstm, predict, lstm_objective
from mlir_bindings import lagrad_lstm_model, lagrad_lstm_predict, lagrad_lstm_objective
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
    assert dweight == pytest.approx(torch_args[0].grad.view(4, -1))
    assert dbias == pytest.approx(torch_args[1].grad.view(4, -1))
    assert dhidden == pytest.approx(torch_args[2].grad)
    assert dcell == pytest.approx(torch_args[3].grad)
    assert dinput == pytest.approx(torch_args[4].grad)


def test_lstm_predict(lstm_input: LSTMInput):
    x = lstm_input.sequence[0]
    torch_args = []
    for i, arg in enumerate(
        [lstm_input.main_params, lstm_input.extra_params, lstm_input.state, x]
    ):
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

    assert dmain == pytest.approx(torch_args[0].grad.view(2, 2, 4, -1))
    assert dextra == pytest.approx(torch_args[1].grad)
    assert dstate == pytest.approx(torch_args[2].grad.view(2, 2, -1))


def test_lstm_objective(lstm_input: LSTMInput):
    seq = lstm_input.sequence[:4]
    state = lstm_input.state.copy()
    torch_args = []
    for i, arg in enumerate(
        [lstm_input.main_params, lstm_input.extra_params, lstm_input.state, seq]
    ):
        targ = torch.from_numpy(arg)
        if i < 3:
            targ.requires_grad = True
        torch_args.append(targ)
    lstm_objective(*torch_args).backward()
    assert lstm_input.state == pytest.approx(state)

    dmain, dextra = lagrad_lstm_objective(
        lstm_input.main_params.reshape(2, 2, 4, -1),
        lstm_input.extra_params,
        lstm_input.state.reshape(2, 2, -1).copy(),
        seq,
    )

    assert dmain == pytest.approx(torch_args[0].grad.view(2, 2, 4, -1))
    assert dextra == pytest.approx(torch_args[1].grad)
