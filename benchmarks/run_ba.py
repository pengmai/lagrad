import argparse
import pandas as pd
import json
from compile import (
    jit_mlir,
    compile_c,
    compile_enzyme,
    compile_mlir,
    link_and_run,
    run_grad,
)
from jinja2 import Environment, PackageLoader, select_autoescape
import numpy as np

np.random.seed(0)
ROUND_TO = 2

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
c_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def main(args):
    config = {
        "nCamParams": 11,
        "n": 49,
        "m": 7776,
        "p": 31843,
        "rot_idx": 0,
        "c_idx": 3,
        "f_idx": 6,
        "x0_idx": 7,
        "rad_idx": 9,
    }
    mlir_template = mlir_env.get_template("ba.mlir")
    # mlir_template = mlir_env.get_template("DELETEME_ba_bufferized.mlir")
    enzyme_template = c_env.get_template("enzyme_ba.c")
    driver_template = c_env.get_template("ba_driver.c")
    helper_template = c_env.get_template("mlir_c_abi.c")
    if args.print:
        print(mlir_template.render(**config))
        return

    compile_mlir(mlir_template.render(**config).encode("utf-8"), "ba_mlir.o")
    compile_enzyme(enzyme_template.render(**config).encode("utf-8"), "enzyme_ba.o")
    compile_c(driver_template.render(**config).encode("utf-8"), "ba_driver.o")
    compile_c(helper_template.render(**config).encode("utf-8"), "mlir_c_abi.o")
    stdout = link_and_run(
        ["ba_mlir.o", "enzyme_ba.o", "ba_driver.o", "mlir_c_abi.o"],
        "ba_driver.out",
        link_runner_utils=True,
    ).decode("utf-8")
    # print(stdout)
    outfile = None
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
        print(results[3:].mean())

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
