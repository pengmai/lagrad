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


def generate_gmm_results(config, outfile=None):
    compile_c(
        config["driver"].render(**config).encode("utf-8"),
        "gmm_driver.o",
    )
    compile_c(
        config["helpers"].render(**config).encode("utf-8"),
        "helpers.o",
    )
    compile_enzyme(config["enzyme"].render(**config).encode("utf-8"), "gmm_enzyme.o")
    compile_mlir(config["mlir"].render(**config).encode("utf-8"), "gmm_kernel.o")
    # compile_mlir(
    #     config["mlir_compressed"].render(**config).encode("utf-8"),
    #     "gmm_compressed_kernel.o",
    # )
    compile_mlir_to_enzyme(
        config["mlir_enzyme"].render(**config).encode("utf-8"),
        output="gmm_mlir_enzyme.o",
        emit="obj",
    )

    stdout = link_and_run(
        [
            "gmm_driver.o",
            "helpers.o",
            "gmm_enzyme.o",
            "gmm_kernel.o",
            # "gmm_compressed_kernel.o",
            "gmm_mlir_enzyme.o",
        ],
        "gmm_driver.out",
        link_runner_utils=True,
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
        if outfile:
            serialized_config = config.copy()
            for template in [
                "driver",
                "helpers",
                "mlir",
                "enzyme",
                "mlir_enzyme",
                "mlir_compressed",
            ]:
                del serialized_config[template]
            # results_dict = {"config": serialized_config, "results": results.to_dict()}
            # with open(outfile, "w") as f:
            #     json.dump(results_dict, f)
        return results
    except (json.JSONDecodeError, Exception):
        print("Failed to parse output")
        print(stdout)


def main(args):
    driver_template = driver_env.get_template("gmm_driver.c")
    helpers_template = driver_env.get_template("mlir_c_abi.c")
    mlir_template = mlir_env.get_template("gmm.mlir")
    mlir_compressed_template = mlir_env.get_template("gmm_compressed.mlir")
    mlir_enzyme_full_template = mlir_env.get_template("gmm_buf_optimized.mlir")
    # mlir_enzyme_template = mlir_env.get_template("gmm_tensor_compressed.mlir")
    # mlir_enzyme_template = mlir_env.get_template("gmm_tensor_bufferized_opt.mlir")

    mlir_enzyme_template = mlir_env.get_template("gmm_buf_compressed.mlir")
    # This is crashing Enzyme for some reason.
    # mlir_enzyme_template = mlir_env.get_template("gmm_buf.mlir")
    enzyme_template = driver_env.get_template("enzyme_gmm.c")
    d, k = 10, 25
    config = {
        "k": k,
        "n": 1000,
        "d": d,
        "tri_size": int(d * (d - 1) / 2),
        "method": "enzyme_mlir_compressed",
        "application": "gmm",
        "driver": driver_template,
        "helpers": helpers_template,
        "mlir": mlir_template,
        "mlir_compressed": mlir_compressed_template,
        "mlir_enzyme": mlir_enzyme_template,
        "mlir_enzyme_full": mlir_enzyme_full_template,
        "enzyme": enzyme_template,
    }
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outfile = osp.join(
        args.results_dir,
        f"{timestamp}_gmm_{config['n']}_{config['k']}_{config['d']}.json",
    )

    if args.print:
        print(mlir_compressed_template.render(**config))
        # print(mlir_template.render(**config))
    else:
        if args.results_file:
            with open(args.results_file) as f:
                results = pd.DataFrame.from_dict(json.load(f)["results"])
        else:
            results = generate_gmm_results(config, outfile)
        if results is not None:
            # primals = results[[col for col in results.columns if "primal" in col]]
            # adjoints = results[[col for col in results.columns if "adjoint" in col]]
            # print(primals[1:].mean())
            print(results.mean())
            # adjoint_means = adjoints[1:].mean()
            # print(adjoint_means)
            # print(
            #     "LAGrad tri adjoint vs enzyme comp adjoint:",
            #     adjoint_means["lagrad_tri_adjoint"]
            #     / adjoint_means["enzyme_comp_adjoint"],
            # )

    # compile_mlir(mlir_template.render(**config).encode("utf-8"), "gmm_kernel.o")
    # rendered = gmm_template.render(
    #     n=n, k=k, d=d, eye=sparse_identity(k), data=data.tolist(), means=means.tolist()
    # )
    # print(jit_mlir(rendered.encode("utf-8"), print_loops=args.loops))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loops",
        "-l",
        action="store_true",
        help="Print the program at the affine loop level",
    )
    parser.add_argument("--results_file", "-r")
    parser.add_argument("--results-dir", default="benchmarks/results")
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print the program without any transformations.",
    )
    args = parser.parse_args()
    main(args)
