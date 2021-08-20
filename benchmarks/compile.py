import subprocess
import os.path as osp

DRIVER_INCLUDES = osp.join(osp.dirname(__file__), "C", "templates")
BIN = osp.join(osp.dirname(__file__), "..", "build", "bin")
TMP = osp.join(osp.dirname(__file__), "tmp")
BUFFERIZE = [
    "-tensor-constant-bufferize",
    "-tensor-bufferize",
    "-linalg-bufferize",
    "-func-bufferize",
    "-finalizing-bufferize",
]
LOWER_TO_LOOPS = ["-convert-linalg-to-loops"]
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


def compile_mlir(contents, output):
    llvm_dialect = run_safe(
        [f"{BIN}/standalone-opt", "-take-grads", "-canonicalize"]
        + BUFFERIZE
        + LOWER_TO_LOOPS
        + LOWER_TO_LLVM,
        stdin=contents,
    )
    llvm_ir = run_safe(["mlir-translate", "-mlir-to-llvmir"], stdin=llvm_dialect)
    obj = run_safe(["llc", "-filetype=obj"], stdin=llvm_ir)
    with open(f"{TMP}/{output}", "wb") as f:
        f.write(obj)


def compile_c(contents, output):
    includes = f"-I{DRIVER_INCLUDES}"
    # The dash tells gcc to read from stdin
    run_safe(
        ["gcc", includes, "-xc", "-c", "-", "-o", f"{TMP}/{output}"], stdin=contents
    )


def link_and_run(objects, binary):
    objects = [f"{TMP}/{obj}" for obj in objects]
    run_safe(["gcc"] + objects + ["-o", f"{TMP}/{binary}"])
    return run_safe([f"{TMP}/{binary}"])
