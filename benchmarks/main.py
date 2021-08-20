import numpy as np
import json
from compile import compile_c, compile_mlir, link_and_run
import argparse
from jinja2 import Environment, PackageLoader, select_autoescape

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
driver_env = Environment(
    loader=PackageLoader("drivers"), autoescape=select_autoescape()
)


def parse_results(results: str):
    """
    The results are printed to stdout with the format:
    BENCHMARK_0 ':' RUNS
    BENCHMARK_1 ':' RUNS
    ...
    BENCHMARK_B ':' RUNS
    """
    parsed = {}
    for line in results.split("\n"):
        [benchmark, runs] = results.split(":")
        parsed[benchmark.strip()] = json.loads(runs)
    return parsed


def main(args):
    N = 131072
    ARGS = [0, 1]
    NUM_WARMUPS = 10
    NUM_RUNS = 100

    kernel_template = mlir_env.get_template("dot.mlir")
    driver_template = driver_env.get_template("dot_driver.c")
    compile_mlir(kernel_template.render(n=N, args=ARGS).encode("utf-8"), "dot_kernel.o")
    compile_c(
        driver_template.render(
            num_warmups=NUM_WARMUPS, num_runs=NUM_RUNS, n=N, args=ARGS
        ).encode("utf-8"),
        "dot_driver.o",
    )
    stdout = link_and_run(["dot_kernel.o", "dot_driver.o"], "dot_driver.out")
    results = parse_results(stdout.decode("utf-8"))
    # print(results)
    print("mean:", np.mean(results["grad_naive"]), 'µs')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
