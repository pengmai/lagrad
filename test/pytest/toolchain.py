"""An entrypoint into the standalone-opt compiler."""

import os.path as osp
import subprocess
from typing import Literal

BIN = osp.join(osp.dirname(__file__), "..", "..", "build", "bin")
MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")
TENSOR_PREPROCESS = ["-convert-elementwise-to-linalg"]
BUFFERIZE = [
    "-tensor-constant-bufferize",
    "-linalg-bufferize",
    "-canonicalize",
    "-func-bufferize",
    # According to the discussion -func-bufferize should be last, but for some
    # reason in this case, -tensor-bufferize must be last.
    "-tensor-bufferize",
]
LOWERING = [
    "-convert-linalg-to-affine-loops",
    "-convert-standalone-to-llvm",
    "-convert-linalg-to-llvm",
    "-convert-std-to-llvm",
    "-llvm-legalize-for-export",
]
ENZYME_DYLIB = osp.join(
    osp.dirname(__file__),
    "..",
    "..",
    "..",
    "playground",
    "enzyme_test",
    "LLVMEnzyme-12.dylib",
)
LIB = osp.expanduser(osp.join("~", ".local", "lib"))
LIBS = osp.join(LIB, "libmlir_runner_utils.dylib")

# Can't run tests in parallel
TMP_DIR = osp.join(osp.dirname(__file__), "tmp")


def lower_to_llvm_dialect(filename: str, take_grads=False) -> bytes:
    try:
        opt_p = subprocess.run(
            [f"{BIN}/standalone-opt", filename]
            + (["-take-grads"] if take_grads else [])
            + TENSOR_PREPROCESS
            + BUFFERIZE
            + LOWERING,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr.decode("utf-8"))
    return opt_p.stdout


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
    enzyme_p = subprocess.run(
        ["opt", "-load", ENZYME_DYLIB, "-enzyme", "-S"],
        input=llvm_ir,
        capture_output=True,
        check=True,
    )
    if optimize:
        opt_p = subprocess.run(
            ["opt", "-O3", "-S"], input=enzyme_p.stdout, capture_output=True, check=True
        )
        return opt_p.stdout
    return enzyme_p.stdout


def compile_pipeline(filename, mode: Literal["enzyme", "grad"] = "enzyme"):
    if mode not in ["enzyme", "grad"]:
        raise ValueError("'mode' must be one of 'enzyme', 'grad'")

    dialect_ir = lower_to_llvm_dialect(filename, take_grads=(mode == "grad"))
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
