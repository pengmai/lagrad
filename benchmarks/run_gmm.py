import argparse
from compile import jit_mlir, compile_c, compile_mlir, link_and_run, run_grad
from jinja2 import Environment, PackageLoader, select_autoescape
import numpy as np

np.random.seed(0)
ROUND_TO = 2

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
driver_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def sparse_identity(k) -> str:
    """
    Return a string representation of an identity matrix that can be template-inserted
    into an MLIR sparse<> attribute.
    """
    eye_indices = "[" + ", ".join([f"[{i}, {i}]" for i in range(k)]) + "]"
    eye_values = f"[{', '.join('1.0' for i in range(k))}]"
    return f"{eye_indices}, {eye_values}"


def compile_and_run_gmm(config):
    compile_c(
        config["driver"].render().encode("utf-8"),
        "gmm_driver.o",
    )
    compile_c(
        config["helpers"].render().encode("utf-8"),
        "helpers.o",
    )
    compile_mlir(config["mlir"].render(**config).encode("utf-8"), "gmm_kernel.o")

    stdout = link_and_run(
        ["gmm_driver.o", "helpers.o", "gmm_kernel.o"],
        "gmm_driver.out",
        link_runner_utils=True,
    )
    print(stdout.decode("utf-8"))


def main(args):
    driver_template = driver_env.get_template("gmm_driver.c")
    helpers_template = driver_env.get_template("mlir_c_abi.c")
    mlir_template = mlir_env.get_template("gmm.mlir")
    config = {
        "k": 25,
        "n": 1000,
        "d": 10,
        "driver": driver_template,
        "helpers": helpers_template,
        "mlir": mlir_template,
    }

    compile_and_run_gmm(config)

    # print(mlir_template.render(**config))

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
    args = parser.parse_args()

    main(args)
