"""
The goal of these benchmarks is to optimize the individual pieces of MLIR generic ops.
"""

from compile import compile_c, compile_mlir, link_and_run
from jinja2 import Environment, PackageLoader, select_autoescape

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
driver_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def main():
    driver_templ = driver_env.get_template("microbenchmark.c")
    helpers_template = driver_env.get_template("mlir_c_abi.c")
    mlir_templ = mlir_env.get_template("microbenchmark.mlir")
    c_templ = driver_env.get_template("microkernels.c")
    config = {"n": 1000, "k": 25, "d": 10}
    compile_mlir(mlir_templ.render(**config).encode("utf-8"), "microbenchmark_mlir.o")
    compile_c(driver_templ.render(**config).encode("utf-8"), "microbenchmark_driver.o")
    compile_c(c_templ.render(**config).encode("utf-8"), "microbench_c.o")
    compile_c(
        helpers_template.render(**config).encode("utf-8"), "microbenchmark_helpers.o"
    )
    output = link_and_run(
        [
            "microbenchmark_mlir.o",
            "microbenchmark_driver.o",
            "microbenchmark_helpers.o",
            "microbench_c.o",
        ],
        "microbenchmark.out",
    )
    print(output.decode("utf-8"))


if __name__ == "__main__":
    main()
