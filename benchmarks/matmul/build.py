#!/usr/bin/env python

import sys
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

if sys.version_info.major < 3:
    print("Python 3 required")
    sys.exit(1)

with new_context() as ctx:
    project = Project("Matrix multiplication micro-benchmark")
    enzyme_sources = glob("*_enzyme.c")
    enzyme_mlir_sources = glob("enzyme*.mlir")
    c_sources = [ffile for ffile in glob("*.c") if ffile not in enzyme_sources]
    lagrad_sources = [
        ffile for ffile in glob("*.mlir") if ffile not in enzyme_mlir_sources
    ]
    c_phase = clang_compile(project, c_sources, include_openblas=True)
    enzyme_phase = compile_enzyme(project, enzyme_sources)
    enzyme_mlir_phase = compile_mlir_enzyme(project, enzyme_mlir_sources)
    lagrad_phase = compile_lagrad(project, lagrad_sources, use_blas=False)
    clang_link(
        project,
        [c_phase, enzyme_phase, enzyme_mlir_phase, lagrad_phase],
        "matmul.out",
        link_openblas=True,
    )
    cli(project)