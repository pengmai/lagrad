import argparse
from compile import (
    compile_mlir_to_enzyme,
    compile_mlir,
    compile_enzyme,
    compile_c,
    link_and_run,
)
from jinja2 import Environment, PackageLoader, select_autoescape

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
c_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def main(args):
    driver_templ = c_env.get_template("lstm_driver.c")
    enzyme_c_templ = c_env.get_template("enzyme_lstm.c")
    helpers_templ = c_env.get_template("mlir_c_abi.c")
    lagrad_templ = mlir_env.get_template("lstm.mlir")

    l, c, b = 2, 1024, 14
    config = {
        "l": l,
        "c": c,
        "b": b,
        "main_sz": 8 * l * b,
        "extra_sz": 3 * b,
        "state_sz": 2 * l * b,
        "seq_sz": c * b,
    }

    if args.print:
        print(lagrad_templ.render(**config))
        return

    compile_mlir(lagrad_templ.render(**config).encode("utf-8"), "lstm_lagrad.o")
    compile_c(driver_templ.render(**config).encode("utf-8"), "lstm_driver.o")
    compile_c(helpers_templ.render(**config).encode("utf-8"), "mlir_c_abi.o")
    compile_enzyme(enzyme_c_templ.render(**config).encode("utf-8"), "lstm_enzyme_c.o")
    stdout = link_and_run(
        ["lstm_driver.o", "mlir_c_abi.o", "lstm_lagrad.o", "lstm_enzyme_c.o"],
        "lstm_driver.out",
        link_runner_utils=True,
    ).decode("utf-8")
    print(stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", "-p", action="store_true")
    args = parser.parse_args()
    main(args)
