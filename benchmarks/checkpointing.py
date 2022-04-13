from compile import (
    compile_c,
    compile_enzyme,
    compile_mlir,
    compile_mlir_to_enzyme,
    link_and_run,
)
from jinja2 import Environment, PackageLoader, select_autoescape

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
c_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def pow_main(args):
    kernel_templ = mlir_env.get_template("pow.mlir")
    driver_templ = c_env.get_template("checkpointing_driver.c")
    helper_templ = c_env.get_template("mlir_c_abi.c")
    compile_mlir(kernel_templ.render().encode("utf-8"), "pow.o")
    compile_c(driver_templ.render().encode("utf-8"), "pow_driver.o")
    compile_c(helper_templ.render().encode("utf-8"), "helpers.o")
    stdout = link_and_run(
        ["pow.o", "pow_driver.o", "helpers.o"],
        "pow.out",
        link_openblas=False,
        link_runner_utils=True,
    )
    print(stdout.decode("utf-8"))


if __name__ == "__main__":
    pow_main({})
