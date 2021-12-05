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

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
c_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def main(args):
    driver_template = c_env.get_template("hand_driver.c")
    helper_template = c_env.get_template("mlir_c_abi.c")
    # enzyme_c_template = c_env.get_template("enzyme_hand_rowmaj.c")
    enzyme_c_template = c_env.get_template("enzyme_hand.c")
    lagrad_template = mlir_env.get_template("hand.mlir")

    config = {
        "nbones": 22,
        "ntheta": 26,
        "nverts": 544,
    }

    if args.print:
        print(lagrad_template.render(**config))
        return

    compile_c(driver_template.render(**config).encode("utf-8"), "hand_driver.o")
    compile_c(helper_template.render(**config).encode("utf-8"), "mlir_c_abi.o")
    compile_enzyme(enzyme_c_template.render(**config).encode("utf-8"), "hand_enzyme.o")
    compile_mlir(lagrad_template.render(**config).encode("utf-8"), "hand_lagrad.o")
    stdout = link_and_run(
        ["hand_driver.o", "mlir_c_abi.o", "hand_enzyme.o", "hand_lagrad.o"],
        "hand_driver.out",
        link_runner_utils=True,
    ).decode("utf-8")
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
