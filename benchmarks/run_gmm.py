import argparse
from compile import (
    compile_c,
    compile_enzyme,
    compile_mlir,
    link_and_run,
)
from jinja2 import Environment, PackageLoader, select_autoescape
import os.path as osp
import json
import pandas as pd

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
driver_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def generate_gmm_results(config, outfile=None):
    compile_c(
        config["driver"].render().encode("utf-8"),
        "gmm_driver.o",
    )
    compile_c(
        config["helpers"].render().encode("utf-8"),
        "helpers.o",
    )
    compile_enzyme(config["enzyme"].render().encode("utf-8"), "gmm_enzyme.o")
    compile_mlir(config["mlir"].render(**config).encode("utf-8"), "gmm_kernel.o")

    stdout = link_and_run(
        ["gmm_driver.o", "helpers.o", "gmm_enzyme.o", "gmm_kernel.o"],
        "gmm_driver.out",
        link_runner_utils=True,
    ).decode("utf-8")

    try:
        lines = stdout.splitlines()
        keys = [
            # "enzyme_full_primal",
            # "enzyme_full_adjoint",
            "enzyme_comp_primal",
            "enzyme_comp_adjoint",
            # "lagrad_full_primal",
            # "lagrad_full_adjoint",
            "lagrad_tri_primal",
            "lagrad_tri_adjoint",
        ]
        assert len(keys) == len(
            lines
        ), f"expected # of apps to match {len(keys)} vs {len(lines)}"
        results = pd.DataFrame.from_dict(
            {key: json.loads(line) for key, line in zip(keys, lines)}
        )
        if outfile:
            serialized_config = config.copy()
            for template in ["driver", "helpers", "mlir", "enzyme"]:
                del serialized_config[template]
            results_dict = {"config": serialized_config, "results": results.to_dict()}
            with open(outfile, "w") as f:
                json.dump(results_dict, f)
        return results
    except json.JSONDecodeError:
        print("Failed to parse output")
        print(stdout)


def main(args):
    driver_template = driver_env.get_template("gmm_driver.c")
    helpers_template = driver_env.get_template("mlir_c_abi.c")
    mlir_template = mlir_env.get_template("gmm.mlir")
    # mlir_template = mlir_env.get_template("gmm_loops_compressed.mlir")
    enzyme_template = driver_env.get_template("enzyme_gmm.c")
    config = {
        "k": 200,
        "n": 1000,
        "d": 128,
        "application": "gmm",
        "driver": driver_template,
        "helpers": helpers_template,
        "mlir": mlir_template,
        "enzyme": enzyme_template,
    }
    outfile = osp.join(
        args.results_dir, f"gmm_{config['n']}_{config['k']}_{config['d']}.json"
    )

    if args.print:
        print(mlir_template.render(**config))
    else:
        with open(outfile) as f:
            results = pd.DataFrame.from_dict(json.load(f)["results"])
        # results = generate_gmm_results(config, outfile)
        if results is not None:
            primals = results[[col for col in results.columns if "primal" in col]]
            adjoints = results[[col for col in results.columns if "adjoint" in col]]
            print(primals[1:].mean())
            adjoint_means = adjoints[1:].mean()
            print(adjoint_means)
            print(
                "LAGrad tri adjoint vs enzyme comp adjoint:",
                adjoint_means["lagrad_tri_adjoint"]
                / adjoint_means["enzyme_comp_adjoint"],
            )

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
    parser.add_argument("--results-dir", default="benchmarks/results")
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print the program without any transformations.",
    )
    args = parser.parse_args()
    main(args)
