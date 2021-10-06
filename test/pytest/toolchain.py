"""An entrypoint into the standalone-opt compiler."""

import os.path as osp
import subprocess
from typing import List, Literal

LLVM_12_BIN = osp.join(osp.expanduser("~"), ".local", "LLVM12", "bin")
OPT_12 = osp.join(LLVM_12_BIN, "opt")
CLANG_12 = osp.join(LLVM_12_BIN, "clang")
LLC_12 = osp.join(LLVM_12_BIN, "llc")
BIN = osp.join(osp.dirname(__file__), "..", "..", "build", "bin")
MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")
TENSOR_PREPROCESS = ["-canonicalize", "-convert-elementwise-to-linalg"]
BUFFERIZE = [
    "-tensor-constant-bufferize",
    "-tensor-bufferize",
    "-linalg-bufferize",
    "-func-bufferize",
    "-finalizing-bufferize",
]
LOWERING = [
    "-convert-linalg-to-loops",
    "-lower-affine",
    "-convert-scf-to-std",
    "-convert-memref-to-llvm",
    "-convert-math-to-llvm",
    "-convert-linalg-to-llvm",
    "-convert-std-to-llvm",
    "-llvm-legalize-for-export",
]
LOWERING_ENZYME = [
    "-convert-linalg-to-loops",
    "-convert-scf-to-std",
    "-convert-memref-to-llvm",
    "-convert-math-to-llvm",
    "-convert-standalone-to-llvm",
    "-convert-linalg-to-llvm",
    "-convert-std-to-llvm",
    "-llvm-legalize-for-export",
]
ENZYME_DYLIB = osp.join(
    osp.expanduser("~"),
    "Research",
    "Enzyme",
    "enzyme",
    "build",
    "Enzyme",
    "LLVMEnzyme-12.dylib",
)
LIB = osp.expanduser(osp.join("~", ".local", "lib"))
LIBS = osp.join(LIB, "libmlir_runner_utils.dylib")

# Can't run tests in parallel
TMP_DIR = osp.join(osp.dirname(__file__), "tmp")


def run_opt(contents: bytes, args: List[str]) -> bytes:
    try:
        opt_p = subprocess.run(
            [f"{BIN}/standalone-opt"] + args,
            input=contents,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr.decode("utf-8"))
    return opt_p.stdout


def lower_to_llvm_dialect(contents: bytes, take_grads=False) -> bytes:
    return run_opt(
        contents,
        (["-take-grads"] if take_grads else [])
        + TENSOR_PREPROCESS
        + BUFFERIZE
        + (LOWERING if take_grads else LOWERING_ENZYME),
    )


def lower_to_llvm(llvm_dialect: bytes) -> bytes:
    translate_p = subprocess.run(
        ["mlir-translate", "-mlir-to-llvmir"],
        input=llvm_dialect,
        capture_output=True,
        check=True,
    )
    return translate_p.stdout


def run_enzyme(llvm_ir: bytes, optimize=True):
    # opt "$LLVM_FILE" -load $ENZYME_DYLIB -enzyme -o "$ENZYME_OUTPUT" -S
    # try:
    #     first_opt = subprocess.run(
    #         [
    #             CLANG_12,
    #             "-S",
    #             "-emit-llvm",
    #             "-xc",
    #             "-",
    #             "-o",
    #             "/dev/stdout",
    #             "-O2",
    #             "-fno-vectorize",
    #             "-fno-slp-vectorize",
    #             "-fno-unroll-loops",
    #         ],
    #         input=llvm_ir,
    #         capture_output=True,
    #         check=True,
    #     )
    # except subprocess.CalledProcessError as e:
    #     raise Exception(e.stderr.decode("utf-8"))
    try:
        enzyme_p = subprocess.run(
            [OPT_12, "-load", ENZYME_DYLIB, "-enzyme", "-S"],
            input=llvm_ir,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr.decode("utf-8"))
    if optimize:
        opt_p = subprocess.run(
            [OPT_12, "-O3", "-S"],
            input=enzyme_p.stdout,
            capture_output=True,
            check=True,
        )
        return opt_p.stdout
    return enzyme_p.stdout


def compile_benchmark(name: str, high_level_ir: bytes):
    llvm_dialect = lower_to_llvm_dialect(high_level_ir, take_grads=False)
    llvm_ir = lower_to_llvm(llvm_dialect)
    object_file = osp.join(TMP_DIR, f"{name}.o")
    with open(object_file, "wb") as ofile:
        subprocess.run(
            ["llc", "-filetype=obj"], input=llvm_ir, stdout=ofile, check=True
        )
    # TODO: Do this


def compile_pipeline(filename, mode: Literal["enzyme", "grad"] = "enzyme"):
    if mode not in ["enzyme", "grad"]:
        raise ValueError("'mode' must be one of 'enzyme', 'grad'")

    with open(filename, "rb") as f:
        contents = f.read()

    dialect_ir = lower_to_llvm_dialect(contents, take_grads=(mode == "grad"))
    llvm_ir = lower_to_llvm(dialect_ir)
    if mode == "enzyme":
        llvm_ir = run_enzyme(llvm_ir, optimize=True)

    object_file = osp.join(TMP_DIR, "app.o")
    with open(object_file, "wb") as ofile:
        subprocess.run(
            ["llc", "-filetype=obj"], input=llvm_ir, stdout=ofile, check=True
        )

    exe_file = osp.join(TMP_DIR, "a.out")
    subprocess.run(["clang", object_file, LIBS, "-o", exe_file], check=True)

    exe_p = subprocess.run(
        [exe_file], env={"DYLD_LIBRARY_PATH": LIB}, capture_output=True
    )
    return exe_p.stdout


def jit_file(filename: str, debug=False) -> str:
    with open(filename, "rb") as f:
        contents = f.read()
    return jit(contents, debug=debug)


def jit(contents: bytes, args=None, debug=False) -> str:
    """
    Execute the given MLIR file through MLIR's JIT, first generalizing any named
    linalg ops to linalg.generic.
    """
    if debug:
        print(
            "\n",
            run_opt(
                contents,
                ["-linalg-generalize-named-ops", "-take-grads", "-canonicalize"],
            ).decode("utf-8"),
        )
    dialect_ir = run_opt(
        contents,
        args
        or [
            "-linalg-generalize-named-ops",
            "-take-grads",
            "-canonicalize",
            "-convert-elementwise-to-linalg",
        ]
        + BUFFERIZE
        + LOWERING,
    )
    try:
        runner = subprocess.run(
            ["mlir-cpu-runner", "-entry-point-result=void", f"-shared-libs={LIBS}"],
            input=dialect_ir,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr.decode("utf-8"))
    return runner.stdout.decode("utf-8")
