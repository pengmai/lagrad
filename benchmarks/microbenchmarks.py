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
from tqdm import tqdm

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
driver_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


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


def loop_main(args):
    driver_templ = driver_env.get_template("loop_driver.c")
    helpers_templ = driver_env.get_template("mlir_c_abi.c")
    enzyme_templ = mlir_env.get_template("nested_loop_enzyme.mlir")
    # mlir_templ = mlir_env.get_template("DELETEME_nested_loop_buf.mlir")
    mlir_templ = mlir_env.get_template("nested_loop_lagrad.mlir")

    config = {"application": "loop", "n": 1024, "k": 512, "d": 512}
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
    config = {"application": "trimatvec", "n": 4096, "method": "tri"}
    assert config["method"] in [
        "full",
        "tri",
        "comp",
    ], f"Unexpected method: {config['method']}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # outfile = osp.join(
    #     args.results_dir,
    #     f"{timestamp}_trmv_{config['n']}.json",
    # )
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


SIZES = {"matmul": [16, 32, 64, 128, 256, 512, 1024]}


def main(args):
    assert args.application in [
        "vecadd",
        "dot",
        "matmul",
    ], f"Unsupported application {args.application}"
    driver_templ = driver_env.get_template(f"microbench_{args.application}_driver.c")
    helpers_template = driver_env.get_template("mlir_c_abi.c")
    mlir_templ = mlir_env.get_template("microbenchmark.mlir")
    enzyme_c_templ = driver_env.get_template("microbenchmark_enzyme.c")
    enzyme_mlir_templ = mlir_env.get_template(
        f"microbenchmarks/{args.application}.mlir"
    )
    c_templ = driver_env.get_template("microkernels.c")
    sizes = SIZES[args.application][-1:]
    final_df = None
    with tqdm(sizes) as t:
        for size in t:
            config = {"n": size}

            if args.print:
                # print(mlir_templ.render(**config))
                print(enzyme_mlir_templ.render(**config))
                return

            compile_mlir(
                mlir_templ.render(**config).encode("utf-8"), "microbenchmark_mlir.o"
            )
            compile_mlir_to_enzyme(
                enzyme_mlir_templ.render(**config).encode("utf-8"),
                "microbenchmark_enzyme_mlir.o",
                emit="obj",
            )
            compile_c(
                driver_templ.render(**config).encode("utf-8"), "microbenchmark_driver.o"
            )
            # compile_c(c_templ.render(**config).encode("utf-8"), "microbench_c.o")
            compile_c(helpers_template.render(**config).encode("utf-8"), "mlir_c_abi.o")
            compile_enzyme(
                enzyme_c_templ.render(**config).encode("utf-8"),
                "microbenchmark_enzyme_c.o",
            )
            stdout = link_and_run(
                [
                    "microbenchmark_mlir.o",
                    "microbenchmark_driver.o",
                    "microbenchmark_enzyme_c.o",
                    "microbenchmark_enzyme_mlir.o",
                    "mlir_c_abi.o",
                ],
                "microbenchmark.out",
            ).decode("utf-8")
            try:
                lines = stdout.splitlines()
                keys = ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]
                assert len(lines) == len(keys), "Unexpected number of lines"
                results = pd.DataFrame.from_dict(
                    {key: json.loads(line) for key, line in zip(keys, lines)}
                )
                results.columns = pd.MultiIndex.from_product(
                    [[config["n"]], results.columns]
                )
                if final_df is None:
                    final_df = results
                else:
                    final_df = final_df.join(results)
            except (json.JSONDecodeError, Exception):
                print("Failed to parse output")
                print(stdout)

    print(final_df.to_csv(sep="\t", index=False))


df_1_txt = """
LAGrad	Enzyme/MLIR	Enzyme/C
223	1145	313
194	373	310
193	372	310
194	372	311
194	372	310
193	372	309
"""
df_2_txt = """
LAGrad	Enzyme/MLIR	Enzyme/C
223	1145	313
194	393	340
193	392	340
194	392	341
194	392	340
193	392	349
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", "-p", action="store_true")
    parser.add_argument("--application", default="vecadd")
    args = parser.parse_args()

    if not args.print:
        print(f"Running application {args.application}")
    # trimatvec_main(args)
    # loop_main(args)
    # cache_main(args)
    main(args)
