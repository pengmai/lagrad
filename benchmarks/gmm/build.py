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
    / "gmm"
    / "data"
    # / "test.txt"
    / "1k"
    / "gmm_d128_K5.txt"
)

with new_context() as ctx:
    with open(data_file, "r") as f:
        d, k, n = [int(x) for x in f.readline().split()]
    template_args = {"n": n, "k": k, "d": d, "tri_size": d * (d - 1) // 2}
    include_path = pathlib.Path(__file__).parent / "include"

    def get_project(title: str, subdir: str, out: str):
        project = Project(title)
        enzyme_sources = glob(f"{subdir}/*_enzyme.c")
        enzyme_mlir_sources = glob(f"{subdir}/enzyme*.mlir")
        c_sources = [
            ffile for ffile in glob(f"{subdir}/*.c") if ffile not in enzyme_sources
        ]
        lagrad_sources = [
            ffile
            for ffile in glob(f"{subdir}/*.mlir")
            if ffile not in enzyme_mlir_sources
        ]
        c_phase = clang_compile(
            project,
            c_sources,
            template_args={"data_file": str(data_file)},
            extra_includes=[include_path],
        )
        enzyme_phase = compile_enzyme(
            project, enzyme_sources, extra_includes=[include_path]
        )
        enzyme_mlir_phase = compile_mlir_enzyme(
            project, enzyme_mlir_sources, template_args
        )
        lagrad_phase = compile_lagrad(project, lagrad_sources, template_args)
        clang_link(
            project,
            [c_phase, enzyme_phase, enzyme_mlir_phase, lagrad_phase],
            f"{out}.out",
        )
        return project

    full_project = get_project("GMM Full", "full", "gmm_full")
    packed_project = get_project("GMM Packed", "packed", "gmm_packed")
    if __name__ == "__main__":
        cli(full_project, packed_project)