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
    / "gmm"
    / "data"
    # / "test.txt"
    / "1k"
    / "gmm_d128_K5.txt"
)


def get_project(title: str, subdir: str, out: str, template_args: dict):
    project = Project(title)
    include_path = pathlib.Path(__file__).parent / "include"
    enzyme_sources = glob(f"{subdir}/*_enzyme.c")
    enzyme_mlir_sources = glob(f"{subdir}/enzyme*.mlir")
    c_sources = [
        ffile for ffile in glob(f"{subdir}/*.c") if ffile not in enzyme_sources
    ]
    lagrad_sources = [
        ffile for ffile in glob(f"{subdir}/*.mlir") if ffile not in enzyme_mlir_sources
    ]
    c_phase = clang_compile(
        project,
        c_sources,
        template_args=template_args,
        extra_includes=[include_path],
        include_openblas=True,
    )
    enzyme_phase = compile_enzyme(
        project, enzyme_sources, extra_includes=[include_path]
    )
    # enzyme_mlir_phase = compile_mlir_enzyme(project, enzyme_mlir_sources, template_args)
    lagrad_phase = compile_lagrad(
        project, lagrad_sources, template_args, use_clang=False
    )
    clang_link(
        project,
        [
            c_phase,
            enzyme_phase,
            # enzyme_mlir_phase,
            lagrad_phase,
        ],
        f"{out}.out",
        link_openblas=True,
    )
    return project


# Enzyme is currently returning incorrect results for the fully-materialized adjoint.
# Going to ignore it for now.
def get_full_project(template_args):
    return get_project("GMM Full", "full", "gmm_full", template_args)


def get_tri_project(template_args):
    return get_project("GMM Tri", "tri", "gmm_tri", template_args)


def get_packed_project(template_args):
    return get_project("GMM Packed", "packed", "gmm_packed", template_args)


if __name__ == "__main__":
    with new_context() as ctx:
        with open(data_file, "r") as f:
            d, k, n = [int(x) for x in f.readline().split()]
        template_args = {
            "n": n,
            "k": k,
            "d": d,
            "tri_size": d * (d - 1) // 2,
            "data_file": str(data_file),
        }

        cli(
            # get_full_project(template_args),
            # get_tri_project(template_args),
            get_packed_project(template_args),
        )
