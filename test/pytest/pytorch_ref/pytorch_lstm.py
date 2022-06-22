import torch


# The LSTM model
def lstm(weight, bias, hidden, cell, _input):
    # NOTE this line came from: gates = hcat(input,hidden) * weight .+ bias
    # for param, name in zip(
    #     (weight, bias, hidden, cell, _input),
    #     ("weight", "bias", "hidden", "cell", "input"),
    # ):
    #     print(f"{name} shape:", param.shape)
    gates = torch.cat((_input, hidden, _input, hidden)) * weight + bias
    hsize = hidden.shape[0]
    forget = torch.sigmoid(gates[0:hsize])
    ingate = torch.sigmoid(gates[hsize : 2 * hsize])
    outgate = torch.sigmoid(gates[2 * hsize : 3 * hsize])
    change = torch.tanh(gates[3 * hsize :])
    cell = cell * forget + ingate * change
    hidden = outgate * torch.tanh(cell)
    return (hidden, cell)


def grad_sigmoid(x, g):
    s = torch.sigmoid(x)
    return g * s * (1 - s)


def grad_tanh(x, g):
    return g / torch.cosh(x) ** 2


# Predict output given an input
def predict(w, w2, s, x):
    s2 = s.clone().detach()
    # NOTE not sure if this should be element-wise or matrix multiplication
    x = x * w2[0]
    for i in range(0, len(s), 2):
        (s2[i], s2[i + 1]) = lstm(w[i], w[i + 1], s[i], s[i + 1], x)
        x = s2[i]
    return (x * w2[1] + w2[2], s2)

# Get the average loss for the LSTM across a sequence of inputs
def lstm_objective(main_params, extra_params, state, sequence, _range=None):
    if _range is None:
        _range = range(0, len(sequence) - 1)

    total = 0.0
    count = 0
    all_states = [state]
    for t in _range:
        ypred, new_state = predict(
            main_params, extra_params, all_states[t], sequence[t]
        )
        all_states.append(new_state)
        ynorm = ypred - torch.log(sum(torch.exp(ypred), 2))
        ygold = sequence[t + 1]
        total += sum(ygold * ynorm)
        count += ypred.shape[0]
    return -total / count


def logsumexp(x):
    return torch.log(sum(torch.exp(x), 2))
