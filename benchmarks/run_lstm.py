import argparse
from compile import (
    compile_mlir_to_enzyme,
    compile_mlir,
    compile_c,
    link_and_run,
)
from jinja2 import Environment, PackageLoader, select_autoescape

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
c_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def main(args):
    driver_templ = c_env.get_template("lstm_driver.c")
    helpers_templ = c_env.get_template("mlir_c_abi.c")
    config = {}
    compile_c(driver_templ.render(**config).encode("utf-8"), "lstm_driver.o")
    compile_c(helpers_templ.render(*config).encode("utf-8"), "mlir_c_abi.o")
    stdout = link_and_run(
        ["lstm_driver.o", "mlir_c_abi.o"], "lstm_driver.out", link_runner_utils=True
    ).decode("utf-8")
    print(stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", "-p", action="store_true")
    args = parser.parse_args()
    main(args)
