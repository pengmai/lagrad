import torch
import timeit
import numpy as np


def run_dot(config):
    x = torch.rand(config["n"], requires_grad=0 in config["args"])
    y = torch.rand(config["n"], requires_grad=1 in config["args"])
    l = torch.dot(x, y)

    def evaluate_grad():
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
        l.backward(retain_graph=True)

    results = (
        np.array(
            timeit.repeat(
                evaluate_grad,
                number=100,
                repeat=config["num_warmups"] + config["num_runs"],
            )
        )
        / 100
        * 1e6
    )
    return results


def run_matvec(config):
    assert config["args"] in [[0], [1], [0, 1]]
    A = torch.rand((config["m"], config["n"]), requires_grad=0 in config["args"])
    x = torch.rand(config["n"], requires_grad=1 in config["args"])
    l = torch.matmul(A, x).sum()

    def evaluate_grad():
        if A.grad is not None:
            A.grad.detach_()
            A.grad.zero_()
        l.backward(retain_graph=True)

    results = (
        np.array(
            timeit.repeat(
                evaluate_grad,
                number=100,
                repeat=config["num_warmups"] + config["num_runs"],
            )
        )  # [config["num_warmups"] :]
        / 100
        * 1e6
    )
    return results


def run_vecmat(config):
    x = torch.rand(config["n"], requires_grad=0 in config["args"])
    A = torch.rand((config["m"], config["n"]), requires_grad=1 in config["args"])
    l = torch.matmul(x, A).sum()

    def evaluate_grad():
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
        l.backward(retain_graph=True)

    results = (
        np.array(
            timeit.repeat(
                evaluate_grad,
                number=100,
                repeat=config["num_warmups"] + config["num_runs"],
            )
        )  # [config["num_warmups"] :]
        / 100
        * 1e6
    )
    return results


def run_matmul(config):
    A = torch.rand((config["m"], config["n"]), requires_grad=0 in config["args"])
    B = torch.rand((config["n"], config["k"]), requires_grad=1 in config["args"])
    l = torch.matmul(A, B).sum()

    def evaluate_grad():
        if A.grad is not None:
            A.grad.detach_()
            A.grad.zero_()
        l.backward(retain_graph=True)

    results = (
        np.array(
            timeit.repeat(
                evaluate_grad,
                number=100,
                repeat=config["num_warmups"] + config["num_runs"],
            )
        )  # [config["num_warmups"] :]
        / 100
        * 1e6
    )
    return results


def run_pytorch(config):
    if config["application"] == "dot":
        return run_dot(config)
    elif config["application"] == "matvec":
        return run_matvec(config)
    elif config["application"] == "vecmat":
        return run_vecmat(config)
    elif config["application"] == "matmul":
        return run_matmul(config)
    raise ValueError(f'Unsupported application {config["application"]}')
