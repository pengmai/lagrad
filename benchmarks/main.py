import numpy as np
import json
from compile import compile_c, compile_mlir, compile_enzyme, link_and_run, OPENBLAS_OBJ
import argparse
from jinja2 import Environment, PackageLoader, select_autoescape

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


def run_dot():
    APPLICATION = "dot"
    render_args = {
        "n": 2 ** 17,
        "application": APPLICATION,
        "args": [0],
        "num_warmups": 10,
        "num_runs": 50,
    }
    # N = 2 ** 17
    # ARGS = [0]
    # NUM_WARMUPS = 10
    # NUM_RUNS = 50

    grad_kernel_template = mlir_env.get_template(f"{APPLICATION}.mlir")
    driver_template = driver_env.get_template("driver.c")
    kernel_template = driver_env.get_template("kernels.c")
    helpers_template = driver_env.get_template("mlir_c_abi.c")
    enzyme_template = driver_env.get_template("enzyme_kernels.c")

    compile_mlir(
        grad_kernel_template.render(prefix="blas_", **render_args).encode("utf-8"),
        f"{APPLICATION}_libcall.o",
        lower_type="blas",
    )
    compile_enzyme(
        enzyme_template.render(**render_args).encode("utf-8"),
        f"{APPLICATION}_enzyme.o",
    )
    compile_mlir(
        grad_kernel_template.render(**render_args).encode("utf-8"),
        f"{APPLICATION}_kernel.o",
    )
    compile_c(
        driver_template.render(**render_args).encode("utf-8"),
        f"{APPLICATION}_driver.o",
    )
    compile_c(helpers_template.render(**render_args).encode("utf-8"), "helpers.o")
    compile_c(kernel_template.render(**render_args).encode("utf-8"), "kernels.o")
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
    results = parse_results(stdout.decode("utf-8"))

    print(f"N: {render_args['n']}")
    for key in results:
        print(f"{key}: {np.mean(results[key])} µs ± {np.std(results[key]):.3} µs")


def main(args):
    run_dot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
