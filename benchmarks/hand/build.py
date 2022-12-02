from ronin.cli import cli
from ronin.projects import Project
from ronin.contexts import new_context
from ronin.utils.paths import glob
from ronin_phases.build_llvm import (
    compile_enzyme,
    compile_mlir_enzyme,
    compile_lagrad,
    clang_compile,
    clang_link,
)
import pathlib

data_file = (
    pathlib.Path.home()
    / "Research"
    / "Enzyme"
    / "enzyme"
    / "benchmarks"
    / "ReverseMode"
    / "hand"
    / "data"
    / "simple_small"
    / "hand1_t26_c100.txt"
    # / "hand12_t26_c100000.txt"
    # / "test.txt"
)


def get_project(template_args: dict):
    project = Project("Hand Tracking")
    enzyme_sources = glob(f"*_enzyme.c")
    enzyme_mlir_sources = glob(f"enzyme*.mlir")
    c_sources = [ffile for ffile in glob(f"*.c") if ffile not in enzyme_sources]
    lagrad_sources = [
        ffile for ffile in glob(f"*.mlir") if ffile not in enzyme_mlir_sources
    ]
    c_phase = clang_compile(project, c_sources)
    enzyme_phase = compile_enzyme(project, enzyme_sources)
    enzyme_mlir_phase = compile_mlir_enzyme(
        project, enzyme_mlir_sources, template_args, replace_hand_opt=True
    )

    # Try to avoid recompiling LAGrad for different datasets
    del template_args["npts"]
    lagrad_phase = compile_lagrad(project, lagrad_sources, template_args)
    clang_link(
        project, [c_phase, enzyme_phase, enzyme_mlir_phase, lagrad_phase], "hand.out"
    )
    return project


if __name__ == "__main__":
    with new_context() as ctx:
        with open(data_file) as f:
            npts, ntheta = [int(x) for x in f.readline().split()]
            assert ntheta == 26, "Unsupported value for ntheta"

        nbones = 22
        template_args = {
            "nbones": nbones,
            "ntheta": 26,
            "nverts": 544,
            "npts": npts,
            "ntriangles": 1084,
        }
        cli(get_project(template_args))
