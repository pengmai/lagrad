import argparse
from compile import (
    compile_c,
    compile_enzyme,
    compile_mlir_to_enzyme,
    compile_mlir,
    link_and_run,
)
from datetime import datetime
from jinja2 import Environment, PackageLoader, select_autoescape
import os.path as osp
import json
import pandas as pd

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
driver_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def adbench_main(args):
    lagrad_templ = mlir_env.get_template("gmm_adbench_lagrad.mlir")
    enzyme_templ = mlir_env.get_template("gmm_adbench_enzyme.mlir")
    driver_templ = driver_env.get_template("gmm_adbench_driver.c")
    helper_templ = driver_env.get_template("mlir_c_abi.c")
    d, k = 128, 200
    config = {
        "k": k,
        "n": 1000,
        "d": d,
    }
    if args.print:
        print(lagrad_templ.render(**config))
        # print(enzyme_templ.render(**config))
        return

    compile_mlir(lagrad_templ.render(**config).encode("utf-8"), "gmm_kernel.o")
    compile_mlir_to_enzyme(
        enzyme_templ.render(**config).encode("utf-8"), "gmm_enzyme.o", emit="obj"
    )
    compile_c(driver_templ.render(**config).encode("utf-8"), "gmm_driver.o")
    compile_c(helper_templ.render(**config).encode("utf-8"), "mlir_c_abi.o")
    stdout = link_and_run(
        ["gmm_kernel.o", "gmm_enzyme.o", "gmm_driver.o", "mlir_c_abi.o"],
        "gmm.out",
        link_runner_utils=True,
    ).decode("utf-8")
    print(stdout)


def main(args):
    driver_template = driver_env.get_template("gmm_driver.c")
    helpers_template = driver_env.get_template("mlir_c_abi.c")
    blas_wrapper_template = driver_env.get_template("kernels_v2.c")
    # mlir_template = mlir_env.get_template("gmm.mlir")
    mlir_template = mlir_env.get_template("gmm_differentiated.mlir")
    mlir_compressed_template = mlir_env.get_template("gmm_compressed.mlir")
    mlir_enzyme_full_template = mlir_env.get_template("gmm_buf_optimized.mlir")
    # mlir_enzyme_template = mlir_env.get_template("gmm_tensor_compressed.mlir")
    # mlir_enzyme_template = mlir_env.get_template("gmm_tensor_bufferized_opt.mlir")

    mlir_enzyme_template = mlir_env.get_template("gmm_buf_compressed.mlir")
    # This is crashing Enzyme for some reason.
    # mlir_enzyme_template = mlir_env.get_template("gmm_buf.mlir")
    enzyme_template = driver_env.get_template("enzyme_gmm.c")
    data_file = "benchmarks/data/gmm_d10_K25.txt"
    with open(data_file) as f:
        d, k, n = [int(x) for x in f.readline().split()]

    config = {
        "k": k,
        "n": n,
        "d": d,
        "tri_size": int(d * (d - 1) / 2),
        "method": "enzyme_mlir_compressed",
        "data_file": "benchmarks/data/gmm_d10_K25.txt",
        "results_file": "benchmarks/data/gmm_results.txt",
    }

    if args.print:
        # print(mlir_compressed_template.render(**config))
        print(mlir_template.render(**config))
        return

    compile_c(
        driver_template.render(**config).encode("utf-8"),
        "gmm_driver.o",
    )
    compile_c(blas_wrapper_template.render().encode("utf-8"), "blas_wrapper.o")
    compile_c(
        helpers_template.render(**config).encode("utf-8"),
        "helpers.o",
    )
    compile_enzyme(enzyme_template.render(**config).encode("utf-8"), "gmm_enzyme.o")
    compile_mlir(mlir_template.render(**config).encode("utf-8"), "gmm_kernel.o")
    # compile_mlir(
    #     mlir_compressed_template.render(**config).encode("utf-8"),
    #     "gmm_compressed_kernel.o",
    # )
    compile_mlir_to_enzyme(
        mlir_enzyme_template.render(**config).encode("utf-8"),
        output="gmm_mlir_enzyme.o",
        emit="obj",
    )

    stdout = link_and_run(
        [
            "gmm_driver.o",
            "helpers.o",
            "blas_wrapper.o",
            "gmm_enzyme.o",
            "gmm_kernel.o",
            "gmm_compressed_kernel.o",
            "gmm_mlir_enzyme.o",
        ],
        "gmm_driver.out",
        link_runner_utils=True,
        link_openblas=True,
    ).decode("utf-8")
    print(stdout)
    return

    try:
        lines = stdout.splitlines()
        keys = [
            # "enzyme_full_primal",
            # "enzyme_full_adjoint",
            # "enzyme_comp_primal",
            # "enzyme_comp_adjoint",
            "mlir_optimized_primal",
            "enzyme_mlir_full_adjoint",
            # "lagrad_full_primal",
            # "lagrad_full_adjoint",
            # "lagrad_tri_primal",
            # "lagrad_tri_adjoint",
            # "lagrad_comp_primal",
            # "lagrad_comp_adjoint",
        ]
        assert len(keys) == len(
            lines
        ), f"expected # of apps to match {len(keys)} vs {len(lines)}"
        results = pd.DataFrame.from_dict(
            {key: json.loads(line) for key, line in zip(keys, lines)}
        )
        print(results.to_csv(sep="\t", index=False))
    except (json.JSONDecodeError, Exception):
        print("Failed to parse output")
        print(stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adbench",
        action="store_true",
        help="Run the ADBench tensorized version of the program",
    )
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print the program without any transformations.",
    )
    args = parser.parse_args()
    if args.adbench:
        adbench_main(args)
    else:
        main(args)
