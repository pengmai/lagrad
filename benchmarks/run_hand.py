import argparse
import pandas as pd
import json
from compile import (
    compile_mlir_to_enzyme,
    jit_mlir,
    compile_c,
    compile_enzyme,
    compile_mlir,
    link_and_run,
    run_grad,
)
from jinja2 import Environment, PackageLoader, select_autoescape
import numpy as np

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
c_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def main(args):
    driver_template = c_env.get_template("hand_driver.c")
    helper_template = c_env.get_template("mlir_c_abi.c")
    # enzyme_c_template = c_env.get_template("enzyme_hand_rowmaj.c")
    enzyme_c_template = c_env.get_template("enzyme_hand.c")
    enzyme_mlir_template = mlir_env.get_template("hand_enzyme.mlir")
    lagrad_template = mlir_env.get_template("hand.mlir")
    # lagrad_template = mlir_env.get_template("hand_inlined.mlir")
    data_file = "benchmarks/data/hand/simple_small/hand1_t26_c100.txt"
    with open(data_file, "r") as f:
        npts, ntheta = [int(x) for x in f.readline().split()]
        assert ntheta == 26, "unsupported value for ntheta"

    nbones = 22
    nverts = 544
    config = {
        "nbones": nbones,
        "ntheta": 26,
        "nverts": nverts,
        "npts": npts,
        "primal_shape": f"{npts}x3",
        "model_dir": "benchmarks/data/hand/simple_small/model",
        "data_file": data_file,
    }

    if args.print:
        # print(enzyme_mlir_template.render(**config))
        print(lagrad_template.render(**config))
        return

    compile_mlir_to_enzyme(
        enzyme_mlir_template.render(**config).encode("utf-8"),
        "hand_enzyme_mlir.o",
        emit="obj",
    )
    compile_c(driver_template.render(**config).encode("utf-8"), "hand_driver.o")
    compile_c(helper_template.render(**config).encode("utf-8"), "mlir_c_abi.o")
    # compile_enzyme(enzyme_c_template.render(**config).encode("utf-8"), "hand_enzyme.o")
    compile_mlir(lagrad_template.render(**config).encode("utf-8"), "hand_lagrad.o")
    stdout = link_and_run(
        [
            "hand_driver.o",
            "mlir_c_abi.o",
            "hand_enzyme.o",
            "hand_lagrad.o",
            "hand_enzyme_mlir.o",
        ],
        "hand_driver.out",
        link_runner_utils=True,
    ).decode("utf-8")

    try:
        lines = stdout.splitlines()
        keys = [
            "lagrad_jacobian",
            "enzyme_jacobian",
        ]
        assert len(keys) == len(
            lines
        ), f"expected # of apps to match {len(keys)} vs {len(lines)}"
        results = pd.DataFrame.from_dict(
            {key: json.loads(line) for key, line in zip(keys, lines)}
        )
        print(results[1:].mean())

        # TODO: Meant to serialize the output results to a file.
        # if outfile:
        #     serialized_config = config.copy()
        #     # results_dict = {"config": serialized_config, "results": results.to_dict()}
        #     # with open(outfile, "w") as f:
        #     #     json.dump(results_dict, f)
        return results
    except (json.JSONDecodeError, Exception):
        print("Failed to parse output")
        print(stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print the result of rendering the MLIR template, then exit.",
    )
    args = parser.parse_args()

    main(args)
