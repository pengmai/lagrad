# Hacks to get access to the toolchain module
import sys

sys.path.append(f"test/pytest")

from toolchain import jit
import argparse
from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())


def main(args):
    template = env.get_template("dot.mlir")
    # print(template.render(n=4, arg=[0]))
    # template = env.get_template("matvec.mlir")
    print(jit(template.render(m=3, n=4, k=5, arg=[0]).encode("utf-8")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
