import subprocess
import os.path as osp

# These must be configured based on the location of OpenBLAS, LLVM version 12, and Enzyme.
OPENBLAS_DIR = osp.join(osp.expanduser("~"), ".local", "OpenBLAS")
LLVM_12_BIN = osp.join(osp.expanduser("~"), "Research", "llvm-project", "build", "bin")
ENZYME_DYLIB = osp.join(
    osp.expanduser("~"),
    "Research",
    "Enzyme",
    "enzyme",
    "build",
    "Enzyme",
    "LLVMEnzyme-12.dylib",
)

DRIVER_INCLUDES = osp.join(osp.dirname(__file__), "C", "templates")
OPENBLAS_INCLUDES = osp.join(OPENBLAS_DIR, "include")
OPENBLAS_OBJ = osp.join(OPENBLAS_DIR, "lib", "libopenblas.a")
CLANG_12 = osp.join(LLVM_12_BIN, "clang-12")
OPT_12 = osp.join(LLVM_12_BIN, "opt")
LLC_12 = osp.join(LLVM_12_BIN, "llc")

BIN = osp.join(osp.dirname(__file__), "..", "build", "bin")
TMP = osp.join(osp.dirname(__file__), "tmp")
BUFFERIZE = [
    "-tensor-constant-bufferize",
    "-tensor-bufferize",
    "-linalg-bufferize",
    "-func-bufferize",
    "-finalizing-bufferize",
]
LOWER_TO_LOOPS = [
    "-convert-linalg-to-affine-loops",
    "-affine-loop-unroll",
    "-lower-affine",
]
LOWER_TO_LIBRARY = ["-convert-linalg-to-std"]
LOWER_TO_LLVM = [
    "-convert-scf-to-std",
    "-convert-memref-to-llvm",
    "-convert-std-to-llvm",
    "-llvm-legalize-for-export",
]


def run_safe(args, stdin: bytes = None):
    try:
        p = subprocess.run(args, input=stdin, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr.decode("utf-8"))
    return p.stdout


def compile_mlir(contents, output, lower_type="loops"):
    assert lower_type in ["loops", "blas"], "Invalid lower_type"

    llvm_dialect = run_safe(
        [f"{BIN}/standalone-opt", "-take-grads", "-canonicalize"]
        + BUFFERIZE
        + (LOWER_TO_LOOPS if lower_type == "loops" else LOWER_TO_LIBRARY)
        + LOWER_TO_LLVM,
        stdin=contents,
    )
    llvm_ir = run_safe(["mlir-translate", "-mlir-to-llvmir"], stdin=llvm_dialect)
    obj = run_safe(["llc", "-filetype=obj"], stdin=llvm_ir)
    with open(f"{TMP}/{output}", "wb") as f:
        f.write(obj)


def compile_enzyme(contents, output):
    includes = f"-I{DRIVER_INCLUDES}"
    preenzyme = run_safe(
        [
            CLANG_12,
            includes,
            "-S",
            "-emit-llvm",
            "-xc",
            "-",
            "-o",
            "/dev/stdout",
            "-O2",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            "-fno-unroll-loops",
        ],
        stdin=contents,
    )
    postenzyme = run_safe(
        [OPT_12, "-S", "-load", ENZYME_DYLIB, "-enzyme", "-O3"], stdin=preenzyme
    )
    run_safe([LLC_12, "-filetype=obj", "-o", f"{TMP}/{output}"], stdin=postenzyme)


def compile_c(contents, output):
    includes = [f"-I{DRIVER_INCLUDES}", f"-I{OPENBLAS_INCLUDES}"]
    # The dash tells gcc to read from stdin
    run_safe(
        ["gcc", "-O3"] + includes + ["-xc", "-c", "-", "-o", f"{TMP}/{output}"],
        stdin=contents,
    )


def link_and_run(objects, binary, link_openblas=True):
    objects = [f"{TMP}/{obj}" for obj in objects] + (
        [OPENBLAS_OBJ] if link_openblas else []
    )
    run_safe(["gcc"] + objects + ["-o", f"{TMP}/{binary}"])
    return run_safe([f"{TMP}/{binary}"])
