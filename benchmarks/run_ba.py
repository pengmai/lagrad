import argparse
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
    driver_template = c_env.get_template("ba_driver.c")
    helper_template = c_env.get_template("mlir_c_abi.c")
    if args.print:
        print(mlir_template.render(**config))
        return

    compile_mlir(mlir_template.render(**config).encode("utf-8"), "ba_mlir.o")
    compile_c(driver_template.render(**config).encode("utf-8"), "ba_driver.o")
    compile_c(helper_template.render(**config).encode("utf-8"), "mlir_c_abi.o")
    output = link_and_run(
        ["ba_mlir.o", "ba_driver.o", "mlir_c_abi.o"],
        "ba_driver.out",
        link_runner_utils=True,
    )
    print(output.decode("utf-8"))


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
