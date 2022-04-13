"""
The goal of these benchmarks is to optimize the individual pieces of MLIR generic ops.
"""

import argparse
import json
import os.path as osp
import pandas as pd
from datetime import datetime
from compile import (
    compile_c,
    compile_enzyme,
    compile_mlir,
    compile_mlir_to_enzyme,
    link_and_run,
)
from jinja2 import Environment, PackageLoader, select_autoescape
import git

repo = git.Repo(search_parent_directories=True)
commit = repo.head.object.hexsha

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
driver_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())
microbench_env = Environment(
    loader=PackageLoader("microbench"), autoescape=select_autoescape()
)


def generate_trimatvec_data(args, config):
    driver_templ = driver_env.get_template("trimatvec_driver.c")
    enzyme_templ = driver_env.get_template("enzyme_kernels.c")
    helpers_templ = driver_env.get_template("mlir_c_abi.c")
    mlir_templ = mlir_env.get_template("trimatvec.mlir")
    enzyme_mlir_templ = mlir_env.get_template("trimatvec_enzyme.mlir")
    # mlir_templ = mlir_env.get_template("trmv.mlir")
    # print(compile_mlir_to_enzyme(enzyme_mlir_templ.render(**config).encode("utf-8")))
    # return

    compile_c(
        driver_templ.render(**config).encode("utf-8"), "trimicrobenchmark_driver.o"
    )
    compile_c(
        helpers_templ.render(**config).encode("utf-8"), "microbenchmark_helpers.o"
    )
    # compile_enzyme(
    #     enzyme_templ.render(**config).encode("utf-8"), "microbenchmark_enzyme.o"
    # )
    compile_mlir(mlir_templ.render(**config).encode("utf-8"), "microbenchmark_mlir.o")
    compile_mlir_to_enzyme(
        enzyme_mlir_templ.render(**config).encode("utf-8"),
        output="microbenchmark_mlir_enzyme.o",
        emit="obj",
    )
    output = link_and_run(
        [
            "trimicrobenchmark_driver.o",
            "microbenchmark_helpers.o",
            # "microbenchmark_enzyme.o",
            "microbenchmark_mlir.o",
            "microbenchmark_mlir_enzyme.o",
        ],
        "trimicrobenchmark_driver.out",
    ).decode("utf-8")

    try:
        lines = output.splitlines()
        keys = [
            # "enzyme_dense_primal",
            # "enzyme_dense_adjoint",
            # "enzyme_tri_primal",
            # "enzyme_tri_adjoint",
            # "enzyme_comp_primal",
            # "enzyme_comp_adjoint",
            # Full
            # "enzyme_mlir_dense_primal",
            # "enzyme_mlir_dense_adjoint",
            # "lagrad_dense_primal",
            # "lagrad_dense_adjoint",
            # Tri
            "enzyme_mlir_tri_primal",
            "enzyme_mlir_tri_adjoint",
            "lagrad_tri_primal",
            "lagrad_tri_adjoint",
            # "lagrad_tri_primal",
            # "lagrad_tri_adjoint",
            # "lagrad_comp_primal",
            # "lagrad_comp_adjoint",
        ]
        if len(lines) != len(keys):
            raise Exception("Computation error")
        results = pd.DataFrame.from_dict(
            {key: json.loads(line) for key, line in zip(keys, lines)}
        )
        return results
    except (json.JSONDecodeError, Exception):
        print("Failed to parse output:")
        print(output)


def hand_err_main(args):
    driver_templ = microbench_env.get_template("hand_err.c")
    enzyme_templ = microbench_env.get_template("hand_err_enzyme.mlir")
    lagrad_templ = microbench_env.get_template("hand_err.mlir")
    helpers_templ = driver_env.get_template("mlir_c_abi.c")

    if args.print:
        print(lagrad_templ.render())
        return

    compile_c(driver_templ.render().encode("utf-8"), "hand_err_driver.o")
    compile_c(helpers_templ.render().encode("utf-8"), "helpers.o")
    compile_mlir_to_enzyme(
        enzyme_templ.render().encode("utf-8"), "hand_err_enzyme.o", emit="obj"
    )
    compile_mlir(lagrad_templ.render().encode("utf-8"), "hand_err_lagrad.o")
    stdout = link_and_run(
        ["hand_err_driver.o", "helpers.o", "hand_err_lagrad.o", "hand_err_enzyme.o"],
        "hand_err.out",
        link_runner_utils=True,
    ).decode("utf-8")
    print(stdout)


def hand_positions_main(args):
    driver_templ = microbench_env.get_template("hand_positions.c")
    lagrad_templ = microbench_env.get_template("hand_positions.mlir")
    helpers_templ = driver_env.get_template("mlir_c_abi.c")
    enzyme_templ = microbench_env.get_template("hand_positions_enzyme.mlir")
    if args.print:
        return
    compile_c(driver_templ.render().encode("utf-8"), "hand_pos_driver.o")
    compile_c(helpers_templ.render().encode("utf-8"), "helpers.o")
    compile_mlir(lagrad_templ.render().encode("utf-8"), "hand_pos_lagrad.o")
    compile_mlir_to_enzyme(
        enzyme_templ.render().encode("utf-8"), "hand_pos_enzyme.o", emit="obj"
    )
    stdout = link_and_run(
        ["hand_pos_driver.o", "helpers.o", "hand_pos_lagrad.o", "hand_pos_enzyme.o"],
        "hand_pos.out",
        link_runner_utils=True,
    ).decode("utf-8")
    print(stdout)


def cache_main(args):
    mlir_templ = mlir_env.get_template("cache_loop.mlir")
    # mlir_templ = mlir_env.get_template("DELETEME_cache_loop_bufferized.mlir")
    enzyme_templ = mlir_env.get_template("cache_loop_enzyme.mlir")
    driver_templ = driver_env.get_template("cache_loop_driver.c")
    helpers_templ = driver_env.get_template("mlir_c_abi.c")
    config = {"n": 22, "d": 1024}
    if args.print:
        print(enzyme_templ.render(**config))
        return
    compile_mlir(mlir_templ.render(**config).encode("utf-8"), "cache_loop_lagrad.o")
    compile_mlir_to_enzyme(
        enzyme_templ.render(**config).encode("utf-8"),
        output="cache_loop_enzyme.o",
        emit="obj",
    )
    compile_c(driver_templ.render(**config).encode("utf-8"), "cache_loop_driver.o")
    compile_c(helpers_templ.render(**config).encode("utf-8"), "cache_loop_helpers.o")
    stdout = link_and_run(
        [
            "cache_loop_driver.o",
            "cache_loop_helpers.o",
            "cache_loop_enzyme.o",
            "cache_loop_lagrad.o",
        ],
        "cache_loop.out",
        link_runner_utils=True,
    ).decode("utf-8")
    print(stdout)
    try:
        lines = stdout.splitlines()
        keys = ["enzyme", "manual_mlir"]
        if len(lines) != len(keys):
            raise Exception("Computation error")
        results = pd.DataFrame.from_dict(
            {key: json.loads(line) for key, line in zip(keys, lines)}
        )
        print(results[2:].mean())
    except (json.JSONDecodeError, Exception):
        print("Failed to parse output")


def main_term_main(args):
    driver_templ = driver_env.get_template("main_term_driver.c")
    helpers_templ = driver_env.get_template("mlir_c_abi.c")
    lagrad_templ = mlir_env.get_template("main_term.mlir")
    # enzyme_templ = mlir_env.get_template("main_term_enzyme.mlir")
    enzyme_c_templ = driver_env.get_template("main_term_kernel.c")
    config = {"application": "loop", "commit": commit, "n": 1000, "k": 200, "d": 128}
    if args.print:
        # print(enzyme_templ.render(**config))
        print(lagrad_templ.render(**config))
        return

    compile_mlir(lagrad_templ.render(**config).encode("utf-8"), "main_term_lagrad.o")
    # compile_mlir_to_enzyme(
    #     enzyme_templ.render(**config).encode("utf-8"), "main_term_enzyme.o", emit="obj"
    # )
    compile_enzyme(enzyme_c_templ.render(**config).encode("utf-8"), "main_term_c.o")
    compile_c(driver_templ.render(**config).encode("utf-8"), "main_term_driver.o")
    compile_c(helpers_templ.render(**config).encode("utf-8"), "helpers.o")
    stdout = link_and_run(
        ["main_term_driver.o", "helpers.o", "main_term_lagrad.o", "main_term_c.o"],
        "main_term.out",
        link_runner_utils=True,
    ).decode("utf-8")
    print(stdout)


def loop_main(args):
    driver_templ = driver_env.get_template("loop_driver.c")
    helpers_templ = driver_env.get_template("mlir_c_abi.c")
    enzyme_templ = mlir_env.get_template("nested_loop_enzyme.mlir")
    # mlir_templ = mlir_env.get_template("DELETEME_nested_loop_buf.mlir")
    mlir_templ = mlir_env.get_template("nested_loop_lagrad.mlir")

    config = {"application": "loop", "commit": commit, "n": 256, "k": 64, "d": 64}
    if args.print:
        print(mlir_templ.render(**config))
        return

    compile_mlir(mlir_templ.render(**config).encode("utf-8"), "loop_lagrad.o")
    compile_mlir_to_enzyme(
        enzyme_templ.render(**config).encode("utf-8"), "loop_enzyme.o", emit="obj"
    )
    compile_c(driver_templ.render(**config).encode("utf-8"), "loop_driver.o")
    compile_c(helpers_templ.render(**config).encode("utf-8"), "loop_helpers.o")
    stdout = link_and_run(
        ["loop_lagrad.o", "loop_enzyme.o", "loop_driver.o", "loop_helpers.o"],
        "loop.out",
        link_runner_utils=True,
    )
    print(stdout.decode("utf-8"))


def trimatvec_main(args):
    config = {"application": "trimatvec", "commit": commit, "n": 4096, "method": "tri"}
    assert config["method"] in [
        "full",
        "tri",
        "comp",
    ], f"Unexpected method: {config['method']}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outfile = osp.join(
        args.results_dir,
        f"{timestamp}_trmv_{config['n']}.json",
    )
    if args.print:
        mlir_templ = mlir_env.get_template("trimatvec_enzyme.mlir")
        print(mlir_templ.render(**config))
        return

    # results = pd.read_csv(osp.join(args.results_dir, "trimicrobench.csv"))
    results = generate_trimatvec_data(args, config)
    if results is not None:
        results_dict = {"config": config, "results": results.to_dict()}
        # with open(outfile, "w") as f:
        #     json.dump(results_dict, f)

        primals = results[[col for col in results.columns if "primal" in col]]
        adjoints = results[[col for col in results.columns if "adjoint" in col]]
        print(primals[10:].mean())
        print(adjoints[10:].mean())


def nestedloop_main(args):
    # config = {"application": "nestedloop"}
    pass


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

    # trimatvec_main(args)
    # loop_main(args)
    # cache_main(args)
    # hand_err_main(args)
    # hand_positions_main(args)
    main_term_main(args)
