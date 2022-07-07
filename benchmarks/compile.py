import warnings
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
MONITOR_BIN = osp.join(
    osp.dirname(__file__), "..", "memory_monitor", "memory_monitor.out"
)

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
    "-buffer-hoisting",
    "-buffer-loop-hoisting",
    "-standalone-loop-hoisting",
    "-promote-buffers-to-stack",
    "-buffer-deallocation",
    "-canonicalize",
]
LOWER_TO_LOOPS = [
    "-convert-linalg-to-loops",
    # "-convert-linalg-to-affine-loops",
    # "-affine-loop-unroll",
    "-lower-affine",
]
LOWER_TO_LIBRARY = ["-convert-linalg-to-std"]
LOWER_TO_LLVM = [
    "-convert-scf-to-std",
    "-convert-memref-to-llvm",
    "-convert-math-to-llvm",
    "-convert-math-to-libm",
    "-convert-std-to-llvm",
    "-reconcile-unrealized-casts",
    "-llvm-legalize-for-export",
]

LOWER_TO_LLVM_WITH_ENZYME = [
    "-convert-scf-to-std",
    "-convert-memref-to-llvm",
    "-convert-math-to-llvm",
    "-convert-math-to-libm",
    "-convert-standalone-to-llvm",
    "-convert-std-to-llvm",
    "-convert-static-allocs",
    "-reconcile-unrealized-casts",
    "-llvm-legalize-for-export",
]


def run_safe(args, stdin: bytes = None, suppress_stderr=False):
    try:
        p = subprocess.run(args, input=stdin, check=True, capture_output=True)
        if p.stderr and not suppress_stderr:
            print(p.stderr.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode("utf-8"))
        raise Exception(e.stderr.decode("utf-8"))
    return p.stdout


def run_grad(contents):
    print(
        run_safe(
            [f"{BIN}/lagrad-opt", "-take-grads", "-canonicalize"],
            stdin=contents,
        ).decode("utf-8")
    )


def compile_mlir(contents, output, lower_type="loops", comprehensive_bufferize=False):
    assert lower_type in ["loops", "blas"], "Invalid lower_type"
    # print(
    #     run_safe(
    #         [
    #             f"{BIN}/lagrad-opt",
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
            f"{BIN}/lagrad-opt",
            # "-loop-invariant-code-motion",
            # "-linalg-generalize-named-ops",
            "-take-grads",
            "-canonicalize",
            "-inline",
            "-linalg-canonicalize",
            # "-standalone-dce",
            "-convert-elementwise-to-linalg",
            "-convert-linalg-triangular-to-loops",
            # "-linalg-fuse-elementwise-ops",
            "-canonicalize",
        ]
        + (
            ["-linalg-comprehensive-module-bufferize=allow-return-memref"]
            if comprehensive_bufferize
            else BUFFERIZE
        )
        + (LOWER_TO_LOOPS if lower_type == "loops" else LOWER_TO_LIBRARY)
        + LOWER_TO_LLVM,
        stdin=contents,
    )
    llvm_ir = run_safe(["mlir-translate", "-mlir-to-llvmir"], stdin=llvm_dialect)
    llvm_ir = run_safe(["opt", "-S", "-O3"], stdin=llvm_ir)
    obj = run_safe(["llc", "-filetype=obj"], stdin=llvm_ir)
    with open(f"{TMP}/{output}", "wb") as f:
        f.write(obj)


def jit_mlir(contents, lower_type="loops", print_loops=False):
    assert lower_type in ["loops", "blas"], "Invalid lower_type"

    if print_loops:
        loops = run_safe(
            [
                f"{BIN}/lagrad-opt",
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
        [f"{BIN}/lagrad-opt", "-convert-elementwise-to-linalg", "-canonicalize"]
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


def compile_mlir_to_enzyme(contents, output="", emit="llvm"):
    def replace_hand_optimization(lines: bytes):
        # warnings.warn(
        #     "Running hand tracking memset_pattern replacement for Enzyme/MLIR"
        # )
        memset_pattern = "@.memset_pattern = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 1.000000e+00, double 1.000000e+00], align 16"
        memset_call = "  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(24) %63, i8* bitcast ([3 x double]* @.memset_pattern to i8*), i64 24, i1 false)"

        def process_line(line: str):
            if line.lstrip().startswith("call void @memset_pattern16"):
                return memset_call
            elif line.startswith("@.memset_pattern ="):
                return memset_pattern
            return line

        return "\n".join(
            [process_line(line) for line in lines.decode("utf-8").splitlines()]
        ).encode("utf-8")

    assert emit in ["llvm", "jit", "obj"], "Invalid emit type"
    llvm_dialect = run_safe(
        [
            f"{BIN}/lagrad-opt",
            # "-linalg-generalize-named-ops",
            "-canonicalize",
            "-convert-elementwise-to-linalg",
            "-convert-linalg-triangular-to-loops",
            "-canonicalize",
        ]
        + BUFFERIZE
        + LOWER_TO_LOOPS
        + LOWER_TO_LLVM_WITH_ENZYME,
        stdin=contents,
    )
    llvm_ir = run_safe(["mlir-translate", "-mlir-to-llvmir"], stdin=llvm_dialect)
    temp_ll_file = osp.join(TMP, "preenzyme.ll")

    pre_ad_opt = True
    if pre_ad_opt:
        with open(temp_ll_file, "w") as f:
            f.write(llvm_ir.decode("utf-8"))
        llvm_ir = run_safe(
            [
                CLANG_12,
                "-S",
                "-emit-llvm",
                temp_ll_file,
                "-o",
                "/dev/stdout",
                "-O2",
                "-fno-vectorize",
                "-fno-slp-vectorize",
                "-ffast-math",
                "-fno-unroll-loops",
            ],
            stdin=llvm_ir,
            suppress_stderr=True,
        )
        # with open(osp.join(TMP, "preenzyme_post_O2.ll"), "wb") as f:
        #     f.write(llvm_ir)
    else:
        warnings.warn("pre-Enzyme optimization disabled")
    # warnings.warn("Using hand-modified post_O2 hand tracking")
    # with open(osp.join(TMP, "preenzyme_post_O2.ll")) as f:
    #     llvm_ir = f.read().encode("utf-8")
    llvm_ir = replace_hand_optimization(llvm_ir)
    postenzyme = run_safe(
        [
            OPT_12,
            "-S",
            "-load",
            ENZYME_DYLIB,
            "-enzyme",
            "-O3",
        ],
        stdin=llvm_ir,
    )

    if emit == "llvm":
        return postenzyme.decode("utf-8")
    elif emit == "jit":
        return run_safe(["lli", "-load", RUNNER_UTILS], stdin=postenzyme).decode(
            "utf-8"
        )
    elif emit == "obj":
        assert output != "", "Output cannot be empty with emit='obj'"
        run_safe([LLC_12, "-filetype=obj", "-o", f"{TMP}/{output}"], stdin=postenzyme)


def compile_enzyme(contents, output, emit="object"):
    def replace_hand_optimization(lines: str):
        warnings.warn("Running hand tracking memset_pattern replacement for Enzyme/C")
        memset_pattern = "@.memset_pattern = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 1.000000e+00, double 1.000000e+00], align 16"
        memset_call = "  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(24) %0, i8* bitcast ([3 x double]* @.memset_pattern to i8*), i64 24, i1 false)"

        def process_line(line: str):
            if line.lstrip().startswith("call void @memset_pattern16"):
                return memset_call
            elif line.startswith("@.memset_pattern ="):
                return memset_pattern
            return line

        return "\n".join([process_line(line) for line in lines.splitlines()])

    assert emit in ["object", "llvm"], "emit must be one of 'object' and 'llvm'"
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

    # with open("main_term_preenzyme_unopt.ll", "wb") as f:
    #     f.write(preenzyme)
    # print("Wrote out preenzyme")
    # preenzyme = replace_hand_optimization(preenzyme.decode("utf-8")).encode("utf-8")
    postenzyme = run_safe(
        [OPT_12, "-S", "-load", ENZYME_DYLIB, "-enzyme", "-O3"], stdin=preenzyme
    )
    if emit == "object":
        run_safe([LLC_12, "-filetype=obj", "-o", f"{TMP}/{output}"], stdin=postenzyme)
    elif emit == "llvm":
        return postenzyme.decode("utf-8")


def compile_c(contents, output):
    includes = [f"-I{DRIVER_INCLUDES}", f"-I{OPENBLAS_INCLUDES}"]
    # The dash tells gcc to read from stdin
    run_safe(
        ["gcc", "-O3", "-Wall"]
        + includes
        + ["-xc", "-c", "-", "-o", f"{TMP}/{output}"],
        stdin=contents,
    )


def link_and_run(
    objects, binary, link_openblas=True, link_runner_utils=False, monitor=False
):
    objects = (
        [f"{TMP}/{obj}" for obj in objects]
        + ([OPENBLAS_OBJ] if link_openblas else [])
        + ([RUNNER_UTILS] if link_runner_utils else [])
    )
    run_safe(
        ["gcc"]
        + objects
        + [
            "-o",
            f"{TMP}/{binary}",
            "-rpath",
            f"{osp.expanduser('~')}/.local/lib",
        ]
    )
    p = subprocess.Popen(
        [f"{TMP}/{binary}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if monitor:
        with open(".secretpass", "rb") as f:
            passwd = f.read()
        usage = run_safe(["sudo", "-S", MONITOR_BIN, str(p.pid)], stdin=passwd).decode(
            "utf-8"
        )
        usage_dict = {
            kv.split(":")[0].strip(): int(kv.split(":")[1].strip())
            for kv in usage.split(",")
        }
    p.wait()
    assert p.returncode == 0, f"Process exited with nonzero exit. stdout: {p.stdout.read()} stderr: {p.stderr.read()}"
    stdout = p.stdout.read()
    if monitor:
        return stdout, usage_dict
    return stdout
