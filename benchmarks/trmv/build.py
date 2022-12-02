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


def get_project(template_args: dict):
    project = Project("TRMV Microbenchmark")
    enzyme_sources = glob(f"*_enzyme.c")
    enzyme_mlir_sources = glob(f"enzyme*.mlir")
    c_sources = [ffile for ffile in glob(f"*.c") if ffile not in enzyme_sources]
    lagrad_sources = [
        ffile for ffile in glob(f"*.mlir") if ffile not in enzyme_mlir_sources
    ]
    c_phase = clang_compile(
        project,
        c_sources,
        template_args=template_args,
    )
    enzyme_phase = compile_enzyme(project, enzyme_sources)
    enzyme_mlir_phase = compile_mlir_enzyme(project, enzyme_mlir_sources, template_args)
    lagrad_phase = compile_lagrad(project, lagrad_sources, template_args)
    clang_link(
        project,
        [
            c_phase,
            enzyme_phase,
            enzyme_mlir_phase,
            lagrad_phase,
        ],
        f"trmv.out",
    )
    return project


if __name__ == "__main__":
    with new_context() as ctx:
        n = 4096
        template_args = {"n": n, "tri_size": n * (n - 1) // 2}
        cli(get_project(template_args))
