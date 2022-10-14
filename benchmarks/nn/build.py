#!/usr/bin/env python

import sys
from ronin.cli import cli
from ronin.contexts import new_context
from ronin.projects import Project
from ronin.utils.paths import glob
from ronin_phases.build_llvm import (
    compile_lagrad,
    compile_enzyme,
    compile_mlir_enzyme,
    clang_compile,
    clang_link,
)

if sys.version_info.major < 3:
    print("Python 3 required")
    sys.exit(1)

with new_context() as ctx:
    project = Project("Multi-layer Perceptron Benchmark")
    enzyme_sources = glob("enzyme_*.c")
    # mlir_sources = glob("DELETEME_postlibrary.mlir")
    enzyme_mlir_sources = glob("enzyme*.mlir")
    mlir_sources = glob("nn.mlir")
    c_sources = [ffile for ffile in glob("*.c") if ffile not in enzyme_sources]

    enzyme_phase = compile_enzyme(project, enzyme_sources)
    enzyme_mlir_phase = compile_mlir_enzyme(project, enzyme_mlir_sources)
    lagrad_phase = compile_lagrad(project, mlir_sources, use_blas=False)
    c_phase = clang_compile(project, c_sources, include_openblas=True)
    clang_link(
        project,
        [c_phase, enzyme_phase, enzyme_mlir_phase, lagrad_phase],
        "neuralnet.out",
        link_openblas=True,
    )
    cli(project)
