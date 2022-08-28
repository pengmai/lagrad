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


def get_project(template_args):
    project = Project("Bundle Adjustment")
    include_path = pathlib.Path(__file__).parent / "include"
    enzyme_sources = glob("*_enzyme.c")
    enzyme_mlir_sources = glob("enzyme*.mlir")
    c_sources = [ffile for ffile in glob("*.c") if ffile not in enzyme_sources]
    lagrad_sources = [
        ffile for ffile in glob("*.mlir") if ffile not in enzyme_mlir_sources
    ]
    c_phase = clang_compile(
        project,
        c_sources,
        extra_includes=[include_path],
    )
    enzyme_phase = compile_enzyme(
        project, enzyme_sources, extra_includes=[include_path]
    )
    enzyme_mlir_phase = compile_mlir_enzyme(project, enzyme_mlir_sources, template_args)
    lagrad_phase = compile_lagrad(project, lagrad_sources, template_args)
    clang_link(
        project,
        [c_phase, enzyme_phase, enzyme_mlir_phase, lagrad_phase],
        f"ba.out",
    )
    return project


if __name__ == "__main__":
    with new_context() as ctx:
        template_args = {
            "nCamParams": 11,
            "rot_idx": 0,
            "c_idx": 3,
            "f_idx": 6,
            "x0_idx": 7,
            "rad_idx": 9,
        }
        project = get_project(template_args)

        cli(project)
