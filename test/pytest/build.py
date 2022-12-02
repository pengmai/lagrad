#!/usr/bin/env python

from ronin.cli import cli
from ronin.projects import Project
from ronin.contexts import new_context
from ronin.utils.paths import glob
from ronin_phases.build_llvm import compile_lagrad, clang_dynamiclib
import pathlib

SRCFILE = pathlib.Path(__file__).parents[1] / "LAGrad"

with new_context() as ctx:
    project = Project("LAGrad tests")

    gmm_sources = [
        abs_path
        for ffile in ["gmm_objective_full"]
        for abs_path in glob(f"{ffile}.mlir", SRCFILE / "gmm")
    ]
    ba_sources = glob("*.mlir", SRCFILE / "ba")
    hand_sources = [
        abs_path
        for ffile in [
            "to_pose_params",
            "get_posed_relatives",
            "relatives_to_absolutes",
            "skinned_vertex_subset",
            "hand_objective",
            "hand_objective_complicated",
        ]
        for abs_path in glob(f"{ffile}.mlir", SRCFILE / "hand")
    ]
    lstm_sources = glob("*.mlir", SRCFILE / "lstm")
    nn_sources = [
        abs_path
        for ffile in ["nn"]
        for abs_path in glob(f"{ffile}.mlir", SRCFILE / "nn")
    ]
    sparse_sources = glob("*.mlir", SRCFILE / "sparse")

    lagrad_phase = compile_lagrad(
        project,
        gmm_sources
        + ba_sources
        + hand_sources
        + lstm_sources
        + nn_sources,
        # + sparse_sources,
        use_clang=False,
        fast_math=False,
    )
    clang_dynamiclib(project, [lagrad_phase], "mlir_bindings.dylib")
    cli(project)
