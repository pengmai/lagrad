import numpy as np
import json
from compile import compile_c, compile_mlir, compile_enzyme, link_and_run, OPENBLAS_OBJ
import argparse
from collections import defaultdict
from tqdm import tqdm
from jinja2 import Environment, PackageLoader, select_autoescape

import matplotlib.pyplot as plt

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
driver_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def parse_results(results: str):
    """
    The results are printed to stdout with the format:
    BENCHMARK_0 ':' RUNS
    BENCHMARK_1 ':' RUNS
    ...
    BENCHMARK_B ':' RUNS
    """
    parsed = {}
    for line in [l for l in results.split("\n") if l]:
        [benchmark, runs] = line.split(":")
        parsed[benchmark.strip()] = json.loads(runs)
    return parsed


def run_application(config):
    APPLICATION = config["application"]
    grad_kernel_template = mlir_env.get_template(f"{APPLICATION}.mlir")
    driver_template = driver_env.get_template(f"{APPLICATION}_driver.c")
    kernel_template = driver_env.get_template("kernels.c")
    helpers_template = driver_env.get_template("mlir_c_abi.c")
    enzyme_template = driver_env.get_template("enzyme_kernels.c")

    compile_mlir(
        grad_kernel_template.render(prefix="blas_", **config).encode("utf-8"),
        f"{APPLICATION}_libcall.o",
        lower_type="blas",
    )
    compile_enzyme(
        enzyme_template.render(**config).encode("utf-8"),
        f"{APPLICATION}_enzyme.o",
    )
    compile_mlir(
        grad_kernel_template.render(**config).encode("utf-8"),
        f"{APPLICATION}_kernel.o",
    )
    compile_c(
        driver_template.render(**config).encode("utf-8"),
        f"{APPLICATION}_driver.o",
    )
    compile_c(helpers_template.render(**config).encode("utf-8"), "helpers.o")
    compile_c(kernel_template.render(**config).encode("utf-8"), "kernels.o")
    stdout = link_and_run(
        [
            f"{APPLICATION}_kernel.o",
            f"{APPLICATION}_libcall.o",
            "helpers.o",
            "kernels.o",
            f"{APPLICATION}_enzyme.o",
            f"{APPLICATION}_driver.o",
        ],
        f"{APPLICATION}_driver.out",
    )
    # print(stdout.decode("utf-8"))
    results = parse_results(stdout.decode("utf-8"))

    return results


def run_matmul():
    sizes = list(range(32, 257, 32))
    raw_results = []
    for size in tqdm(sizes):
        config = {
            "m": size,
            "n": size,
            "k": size,
            "application": "matmul",
            "args": [0],
            "num_warmups": 10,
            "num_runs": 10,
        }
        raw_results.append(run_application(config))

    with open("matmul_results.json", "w") as f:
        json.dump(raw_results, f)
    # with open("matmul_results.json", "r") as f:
    #     raw_results = json.load(f)
    plot_results(
        raw_results,
        sizes=sizes,
        title="(NxN) by (NxN) Matmul Gradient Performance (lower is better)",
        xlabel="N",
    )


def run_matvec():
    sizes = list(range(256, 1025, 128))
    raw_results = []
    for size in tqdm(sizes):
        config = {
            "n": size,
            "m": size,
            "application": "matvec",
            "args": [0],
            "num_warmups": 10,
            "num_runs": 50,
        }
        raw_results.append(run_application(config))

    with open("matvec_results.json", "w") as f:
        json.dump(raw_results, f)
    # with open("matvec_results.json", "r") as f:
    #     raw_results = json.load(f)
    plot_results(
        raw_results,
        sizes=sizes,
        title="(NxN) by N Matrix-Vector Multiplication Gradient Performance (lower is better)",
        xlabel="N",
    )


def run_dot():
    sizes = np.array(list(range(1024, 2 ** 20, 4096 * 10)))
    raw_results = []
    for n in tqdm(sizes):
        config = {
            "n": n,
            "application": "dot",
            "args": [0],
            "num_warmups": 10,
            "num_runs": 50,
        }
        raw_results.append(run_application(config))

    with open("dot_results.json", "w") as f:
        json.dump(raw_results, f)
    plot_results(
        raw_results, sizes=sizes, title="Dot Gradient Performance (lower is better)"
    )


def plot_results(
    raw_results, sizes=[], title="", xlabel="Size of arrays", ylabel="Runtime (µs)"
):
    results = {"means": defaultdict(list), "stds": defaultdict(list)}
    # with open("dot_results.json", "r") as f:
    #     raw_results = json.load(f)
    for i, run in enumerate(raw_results):
        for method in run:
            results["means"][method].append(np.mean(run[method]))
            results["stds"][method].append(np.std(run[method]))

    for method in results["means"]:
        plt.plot(sizes, results["means"][method], label=method)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()


def main(args):
    run_matvec()
    # for key in results:
    #     print(f"{key}: {np.mean(results[key])} µs ± {np.std(results[key]):.3} µs")

    return
    run_dot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
