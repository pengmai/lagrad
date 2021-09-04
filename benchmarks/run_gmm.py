import argparse
from compile import jit_mlir
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loops",
        "-l",
        action="store_true",
        help="Print the program at the affine loop level",
    )
    args = parser.parse_args()

    gmm_template = mlir_env.get_template("gmm.mlir")
    k = 3
    n = 2
    d = 4

    data = np.around(np.random.rand(n, d), ROUND_TO)
    means = np.around(np.random.rand(k, d), ROUND_TO)

    print("data:\n", data)
    print("means:\n", means)
    print("centered:\n", np.stack([data[i] - means for i in range(n)]))
    rendered = gmm_template.render(
        n=n, k=k, d=d, eye=sparse_identity(k), data=data.tolist(), means=means.tolist()
    )
    print(jit_mlir(rendered.encode("utf-8"), print_loops=args.loops))
