"""
The goal of these benchmarks is to optimize the individual pieces of MLIR generic ops.
"""

import argparse
import json
import os.path as osp
import pandas as pd
from compile import compile_c, compile_enzyme, compile_mlir, link_and_run
from jinja2 import Environment, PackageLoader, select_autoescape

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
driver_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def generate_trimatvec_data(args):
    driver_templ = driver_env.get_template("trimatvec_driver.c")
    enzyme_templ = driver_env.get_template("enzyme_kernels.c")
    helpers_templ = driver_env.get_template("mlir_c_abi.c")
    mlir_templ = mlir_env.get_template("trimatvec.mlir")
    config = {"application": "trimatvec", "n": 1024}

    if args.print:
        print(mlir_templ.render(**config))
        return

    compile_c(
        driver_templ.render(**config).encode("utf-8"), "trimicrobenchmark_driver.o"
    )
    compile_c(
        helpers_templ.render(**config).encode("utf-8"), "microbenchmark_helpers.o"
    )
    compile_enzyme(
        enzyme_templ.render(**config).encode("utf-8"), "microbenchmark_enzyme.o"
    )
    compile_mlir(mlir_templ.render(**config).encode("utf-8"), "microbenchmark_mlir.o")
    output = link_and_run(
        [
            "trimicrobenchmark_driver.o",
            "microbenchmark_helpers.o",
            "microbenchmark_enzyme.o",
            "microbenchmark_mlir.o",
        ],
        "trimicrobenchmark_driver.out",
    ).decode("utf-8")

    try:
        lines = output.splitlines()
        keys = [
            "enzyme_dense_primal",
            "enzyme_dense_adjoint",
            "enzyme_tri_primal",
            "enzyme_tri_adjoint",
            "enzyme_comp_primal",
            "enzyme_comp_adjoint",
            "lagrad_dense_primal",
            "lagrad_dense_adjoint",
            "lagrad_tri_primal",
            "lagrad_tri_adjoint",
            "lagrad_comp_primal",
            "lagrad_comp_adjoint",
        ]
        results = pd.DataFrame.from_dict(
            {key: json.loads(line) for key, line in zip(keys, lines)}
        )
        results.to_csv(osp.join(args.results_dir, "trimicrobench.csv"), index=False)
        print(results[10:].mean())
    except json.JSONDecodeError:
        print("Failed to parse output:")
        print(output)


def trimatvec_main(args):
    # generate_trimatvec_data(args)
    results = pd.read_csv(osp.join(args.results_dir, "trimicrobench.csv"))
    primals = results[[col for col in results.columns if "primal" in col]]
    adjoints = results[[col for col in results.columns if "adjoint" in col]]
    print(primals[10:].mean())
    print(adjoints[10:].mean())
    results_dict = results.to_dict()
    print(results_dict)
    # with open(osp.join(args.results_dir, "trimicrobench.json")) as f:
    #     pass


def main():
    driver_templ = driver_env.get_template("microbenchmark.c")
    helpers_template = driver_env.get_template("mlir_c_abi.c")
    mlir_templ = mlir_env.get_template("microbenchmark.mlir")
    c_templ = driver_env.get_template("microkernels.c")
    config = {"n": 1000, "k": 25, "d": 10}
    compile_mlir(mlir_templ.render(**config).encode("utf-8"), "microbenchmark_mlir.o")
    compile_c(driver_templ.render(**config).encode("utf-8"), "microbenchmark_driver.o")
    compile_c(c_templ.render(**config).encode("utf-8"), "microbench_c.o")
    compile_c(
        helpers_template.render(**config).encode("utf-8"), "microbenchmark_helpers.o"
    )
    output = link_and_run(
        [
            "microbenchmark_mlir.o",
            "microbenchmark_driver.o",
            "microbenchmark_helpers.o",
            "microbench_c.o",
        ],
        "microbenchmark.out",
    )
    print(output.decode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="benchmarks/results")
    parser.add_argument("--print", "-p", action="store_true")
    args = parser.parse_args()

    trimatvec_main(args)
