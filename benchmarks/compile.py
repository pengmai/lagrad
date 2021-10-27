import subprocess
import os.path as osp

# These must be configured based on the location of OpenBLAS, LLVM version 12, and Enzyme.
OPENBLAS_DIR = osp.join(osp.expanduser("~"), ".local", "OpenBLAS")
LLVM_12_BIN = osp.join(osp.expanduser("~"), ".local", "LLVM12", "bin")
ENZYME_DYLIB = osp.join(
    osp.expanduser("~"),
    "Research",
    "Enzyme",
    "enzyme",
    "build",
    "Enzyme",
    "LLVMEnzyme-12.dylib",
)
RUNNER_UTILS = osp.join(
    osp.expanduser("~"), ".local", "lib", "libmlir_runner_utils.dylib"
)

SYSTEM_INCLUDES = osp.join(
    "/Applications",
    "Xcode.app",
    "Contents",
    "Developer",
    "Platforms",
    "MacOSX.platform",
    "Developer",
    "SDKs",
    "MacOSX.sdk",
    "usr",
    "include",
)
# Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include
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
    "-standalone-bufferize",
    "-linalg-bufferize",
    "-scf-bufferize",
    "-func-bufferize",
    "-finalizing-bufferize",
    "-buffer-deallocation",
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
    "-convert-math-to-llvm",
    "-convert-std-to-llvm",
    "-reconcile-unrealized-casts",
    "-llvm-legalize-for-export",
]


def run_safe(args, stdin: bytes = None, suppress_stderr=False):
    try:
        p = subprocess.run(args, input=stdin, check=True, capture_output=True)
        if p.stderr and not suppress_stderr:
            print(p.stderr.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr.decode("utf-8"))
    return p.stdout


def run_grad(contents):
    print(
        run_safe(
            [f"{BIN}/standalone-opt", "-take-grads", "-canonicalize"],
            stdin=contents,
        ).decode("utf-8")
    )


def compile_mlir(contents, output, lower_type="loops"):
    assert lower_type in ["loops", "blas"], "Invalid lower_type"
    # print(
    #     run_safe(
    #         [
    #             f"{BIN}/standalone-opt",
    #             "-take-grads",
    #             "-linalg-generalize-named-ops",
    #             "-canonicalize",
    #             "-convert-elementwise-to-linalg",
    #             "-convert-linalg-triangular-to-loops",
    #             "-canonicalize",
    #         ] + BUFFERIZE + ["-canonicalize"],
    #         stdin=contents,
    #     ).decode("utf-8")
    # )

    llvm_dialect = run_safe(
        [
            f"{BIN}/standalone-opt",
            "-take-grads",
            # "-linalg-generalize-named-ops",
            "-canonicalize",
            "-convert-elementwise-to-linalg",
            "-convert-linalg-triangular-to-loops",
        ]
        + BUFFERIZE
        + (LOWER_TO_LOOPS if lower_type == "loops" else LOWER_TO_LIBRARY)
        + LOWER_TO_LLVM,
        stdin=contents,
    )
    llvm_ir = run_safe(["mlir-translate", "-mlir-to-llvmir"], stdin=llvm_dialect)
    llvm_ir = run_safe(["opt", "-O3"], stdin=llvm_ir)
    obj = run_safe(["llc", "-filetype=obj"], stdin=llvm_ir)
    with open(f"{TMP}/{output}", "wb") as f:
        f.write(obj)


def jit_mlir(contents, lower_type="loops", print_loops=False):
    assert lower_type in ["loops", "blas"], "Invalid lower_type"

    if print_loops:
        loops = run_safe(
            [
                f"{BIN}/standalone-opt",
                "-take-grads",
                "-convert-elementwise-to-linalg",
                "-canonicalize",
            ]
            + BUFFERIZE
            + ["-convert-linalg-to-affine-loops"],
            stdin=contents,
        )
        print(loops.decode("utf-8"))

    llvm_dialect = run_safe(
        [f"{BIN}/standalone-opt", "-convert-elementwise-to-linalg", "-canonicalize"]
        + BUFFERIZE
        + (LOWER_TO_LOOPS if lower_type == "loops" else LOWER_TO_LIBRARY)
        + LOWER_TO_LLVM,
        stdin=contents,
    )
    output = run_safe(
        ["mlir-cpu-runner", "-entry-point-result=void", f"-shared-libs={RUNNER_UTILS}"],
        stdin=llvm_dialect,
    )
    return output.decode("utf-8")


def compile_enzyme(contents, output):
    includes = [f"-I{DRIVER_INCLUDES}", f"-I{SYSTEM_INCLUDES}"]
    preenzyme = run_safe(
        [CLANG_12]
        + includes
        + [
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
        suppress_stderr=True,
    )
    postenzyme = run_safe(
        [OPT_12, "-S", "-load", ENZYME_DYLIB, "-enzyme", "-O3"], stdin=preenzyme
    )
    run_safe([LLC_12, "-filetype=obj", "-o", f"{TMP}/{output}"], stdin=postenzyme)


def compile_c(contents, output):
    includes = [f"-I{DRIVER_INCLUDES}", f"-I{OPENBLAS_INCLUDES}"]
    # The dash tells gcc to read from stdin
    run_safe(
        ["gcc", "-O3", "-Wall"]
        + includes
        + ["-xc", "-c", "-", "-o", f"{TMP}/{output}"],
        stdin=contents,
    )


def link_and_run(objects, binary, link_openblas=True, link_runner_utils=False):
    objects = (
        [f"{TMP}/{obj}" for obj in objects]
        + ([OPENBLAS_OBJ] if link_openblas else [])
        + ([RUNNER_UTILS] if link_runner_utils else [])
    )
    run_safe(
        ["gcc"]
        + objects
        + ["-o", f"{TMP}/{binary}", "-rpath", f"{osp.expanduser('~')}/.local/lib"]
    )
    return run_safe([f"{TMP}/{binary}"])
