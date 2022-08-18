from ronin.gcc import GccExecutor, GccCompile, GccLink
from ronin.executors import ExecutorWithArguments
from ronin.phases import Phase
from ronin.projects import Project
from ronin.utils.platform import which
import pathlib
import subprocess

ENZYME_DYLIB = (
    pathlib.Path.home()
    / "Research"
    / "Enzyme"
    / "enzyme"
    / "build"
    / "Enzyme"
    / "LLVMEnzyme-12.dylib"
)

LAGRAD_BINARY = str(pathlib.Path(__file__).parents[1] / "build" / "bin" / "lagrad-opt")

LAGRAD_LLVM_DYLIB = (
    pathlib.Path.home()
    / "Research"
    / "profile-pass"
    / "build"
    / "profiler"
    / "libProfilerPass.dylib"
)

LOCAL_LIB = pathlib.Path.home() / ".local" / "lib"
LOCAL_INCLUDE = pathlib.Path.home() / ".local" / "include"
LAGRAD_UTILS = LOCAL_LIB / "lagrad_utils.dylib"
MLIR_RUNNER_UTILS = LOCAL_LIB / "libmlir_runner_utils.dylib"


class LAGradOptFlags:
    preprocess = [
        "-take-grads",
        "-canonicalize",
        "-inline",
        "-linalg-canonicalize",
        "-pack-triangular",
        "-standalone-dce",
        "-symbol-dce",
        "-convert-elementwise-to-linalg",
        "-convert-linalg-triangular-to-loops",
        "-canonicalize",
    ]
    bufferize = [
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
    lower_to_llvm = [
        # "-convert-linalg-to-library",
        "-convert-linalg-to-loops",
        # "-convert-linalg-to-affine-loops",
        # "-affine-loop-unroll",
        "-lower-affine",
        "-convert-scf-to-std",
        "-convert-memref-to-llvm",
        "-convert-math-to-llvm",
        "-convert-math-to-libm",
        "-convert-std-to-llvm",
        "-reconcile-unrealized-casts",
        "-llvm-legalize-for-export",
    ]
    lower_to_llvm_with_enzyme = [
        "-convert-linalg-to-loops",
        "-lower-affine",
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


PRE_ENZYME_OPT = [
    "-O2",
    "-fno-vectorize",
    "-fno-slp-vectorize",
    "-ffast-math",
    "-fno-unroll-loops",
]


class ClangEmitLLVM(GccExecutor):
    def __init__(self, command: str = None, ccache=True, platform=None, makefile=True):
        super(ClangEmitLLVM, self).__init__(command, ccache, platform)
        self.command = lambda ctx: which(
            ctx.fallback(command, "gcc.gcc_command", "clang-12")
        )
        self.add_argument_unfiltered("$in")
        self.add_argument("-S", "-emit-llvm")

        self.command_types = ["clang_compile"]
        self.output_type = "object"
        self.output_extension = "ll"
        if makefile:
            self.create_makefile_ignore_system()
            self.add_argument_unfiltered("-MF", "$out.d")  # set_makefile_path
            self._deps_file = "$out.d"
            self._deps_type = "gcc"

    def ignore_override_module(self):
        self.add_argument("-Wno-override-module")
        return self


class ClangCompileLLVM(GccExecutor):
    def __init__(self, command: str = None, ccache=True, platform=None):
        super(ClangCompileLLVM, self).__init__(command, ccache, platform)
        self.command = lambda ctx: which(
            ctx.fallback(command, "gcc.gcc_command", "clang-12")
        )
        self.add_argument_unfiltered("$in")

        self.command_types = ["clang_compile"]
        self.output_type = "object"
        self.output_extension = "o"
        self.compile_only()

    def ignore_override_module(self):
        self.add_argument("-Wno-override-module")
        return self


class RenderTemplateExecutor(ExecutorWithArguments):
    def __init__(self, output_extension="mlir"):
        super(RenderTemplateExecutor, self).__init__()
        self.command = lambda ctx: str(pathlib.Path(__file__).parent / "render.py")
        self.add_argument_unfiltered("$in")
        self.add_argument_unfiltered("-o", "$out")
        self.output_type = "object"
        self.output_prefix = "rendered_"
        self.output_extension = output_extension

    def with_args(self, template_args: dict[str, int]):
        for k, v in template_args.items():
            self.add_argument(f"-{k}", v)
        return self


class LAGradOptExecutor(ExecutorWithArguments):
    def __init__(self, command: str = None, default_args=True):
        super(LAGradOptExecutor, self).__init__()
        self.command = lambda ctx: which(
            ctx.fallback(command, "lagrad.opt_command", "lagrad-opt")
        )
        self.add_argument_unfiltered("$in")
        self.add_argument_unfiltered("-o", "$out")
        self.output_type = "object"
        self.output_extension = "mlir"
        if default_args:
            self.add_argument(*LAGradOptFlags.preprocess)
            self.add_argument(*LAGradOptFlags.bufferize)
            self.add_argument(*LAGradOptFlags.lower_to_llvm)


class MLIRTranslateExecutor(ExecutorWithArguments):
    def __init__(self, command: str = None):
        super(MLIRTranslateExecutor, self).__init__()
        self.command = lambda ctx: which(
            ctx.fallback(command, "lagrad.translate_command", "mlir-translate")
        )
        self.add_argument_unfiltered("$in")
        self.add_argument_unfiltered("-o", "$out")
        self.output_type = "object"
        self.output_extension = "ll"
        self.add_argument("-mlir-to-llvmir")


class LLCExecutor(ExecutorWithArguments):
    def __init__(self):
        super(LLCExecutor, self).__init__()
        self.command = lambda ctx: which("llc")
        self.add_argument_unfiltered("$in")
        self.add_argument_unfiltered("-o", "$out")
        self.output_type = "object"
        self.output_extension = "o"
        self.add_argument("-filetype=obj")


class OptExecutor(ExecutorWithArguments):
    def __init__(self, command: str = None, asm=False):
        super(OptExecutor, self).__init__()
        self.command = lambda ctx: which(
            ctx.fallback(command, "llvm.opt_command", "opt")
        )
        self.add_argument_unfiltered("$in")
        self.add_argument_unfiltered("-o", "$out")
        self.output_type = "object"
        self.output_extension = "ll"
        if asm:
            self.add_argument("-S")

    def run_enzyme(self):
        self.add_argument("-load", ENZYME_DYLIB, "-enzyme")
        return self

    def run_fast_math(self):
        self.add_argument(
            "-enable-new-pm=0", "-load", LAGRAD_LLVM_DYLIB, "-llfast-math"
        )
        return self


def compile_enzyme(project: Project, inputs: list[str]) -> Phase:
    emit_llvm = ClangEmitLLVM()
    emit_llvm.add_argument(*PRE_ENZYME_OPT)
    emit_llvm.add_include_path(LOCAL_INCLUDE)
    preenzyme = Phase(
        project=project,
        name="Pre-Enzyme C to LLVM IR",
        inputs=inputs,
        executor=emit_llvm,
    )

    run_enzyme = OptExecutor("opt-12").run_enzyme()
    postenzyme = Phase(
        project=project,
        name="Enzyme AD",
        inputs_from=[preenzyme],
        executor=run_enzyme,
    )

    enzyme_clang = ClangCompileLLVM()
    enzyme_clang.optimize(3)
    return Phase(
        project=project,
        name="Post-Enzyme Objects",
        inputs_from=[postenzyme],
        executor=enzyme_clang,
    )


def compile_mlir_enzyme(
    project: Project,
    inputs: list[str],
    template_args: dict[str, int] = None,
    pre_ad_opt=True,
) -> Phase:
    lagrad_opt = LAGradOptExecutor(default_args=False)
    lagrad_opt.add_argument(*LAGradOptFlags.bufferize)
    lagrad_opt.add_argument(*LAGradOptFlags.lower_to_llvm_with_enzyme)
    llvm_dialect_kwargs = {
        "project": project,
        "name": "Enzyme MLIR Lower to LLVM Dialect",
        "executor": lagrad_opt,
    }

    if template_args:
        llvm_dialect_kwargs["inputs_from"] = [
            Phase(
                project=project,
                name="Render Enzyme MLIR",
                inputs=inputs,
                executor=RenderTemplateExecutor().with_args(template_args),
            )
        ]
    else:
        llvm_dialect_kwargs["inputs"] = inputs

    llvm_dialect = Phase(**llvm_dialect_kwargs)

    llvm_ir = Phase(
        project=project,
        name="Enzyme MLIR Lower to LLVM IR",
        inputs_from=[llvm_dialect],
        executor=MLIRTranslateExecutor(),
    )
    if pre_ad_opt:
        preenzyme_clang = ClangEmitLLVM(makefile=False)
        preenzyme_clang.add_argument(*PRE_ENZYME_OPT)
        preenzyme_clang.ignore_override_module()
        llvm_ir = Phase(
            project=project,
            name="Enzyme MLIR Pre-AD Optimization",
            inputs_from=[llvm_ir],
            executor=preenzyme_clang,
        )

    postenzyme = Phase(
        project=project,
        name="Enzyme MLIR AD",
        inputs_from=[llvm_ir],
        executor=OptExecutor("opt-12").run_enzyme(),
    )

    enzyme_clang = ClangCompileLLVM()
    enzyme_clang.optimize(3)
    return Phase(
        project=project,
        name="MLIR Post-Enzyme Objects",
        inputs_from=[postenzyme],
        executor=enzyme_clang,
    )


def get_sdk_root():
    p = subprocess.run(["xcrun", "--show-sdk-path"], check=True, capture_output=True)
    return p.stdout.decode("utf-8")


def compile_lagrad(
    project: Project,
    inputs: list[str],
    template_args: dict[str, int] = None,
    fast_math=True,
    use_clang=True,
) -> Phase:
    lower_llvm_kwargs = {
        "project": project,
        "name": "Lower to LLVM Dialect",
        "executor": LAGradOptExecutor(),
        "rebuild_on": [LAGRAD_BINARY],
    }
    if template_args:
        lower_llvm_kwargs["inputs_from"] = [
            Phase(
                project=project,
                name="Render Template",
                inputs=inputs,
                executor=RenderTemplateExecutor().with_args(template_args),
            )
        ]
    else:
        lower_llvm_kwargs["inputs"] = inputs
    lagrad_llvm_dialect = Phase(**lower_llvm_kwargs)
    llvm_ir = Phase(
        project=project,
        name="LAGrad Translate to LLVM IR",
        inputs_from=[lagrad_llvm_dialect],
        executor=MLIRTranslateExecutor(),
    )
    if fast_math:
        llvm_ir = Phase(
            project=project,
            name="LAGrad fast-math flags",
            inputs_from=[llvm_ir],
            executor=OptExecutor().run_fast_math(),
        )
    if use_clang:
        lagrad_clang = ClangCompileLLVM()
        lagrad_clang.optimize(3)
        lagrad_clang.ignore_override_module()
        return Phase(
            project=project,
            name="Compile LAGrad objects",
            inputs_from=[llvm_ir],
            executor=lagrad_clang,
        )
    else:
        return Phase(
            project=project,
            name="LLC Compile LAGrad objects",
            inputs_from=[llvm_ir],
            executor=LLCExecutor(),
        )


def clang_compile(
    project: Project, inputs: list[str], template_args: dict[str, int] = None
):
    compile_executor = GccCompile("clang-12")
    compile_executor.optimize(3)
    compile_executor.add_include_path(LOCAL_INCLUDE)

    kwargs = {"project": project, "name": "Compile C", "executor": compile_executor}
    if template_args:
        # Need to include the file's directory if rendering a template
        for parent in set([pathlib.Path(ffile).parent for ffile in inputs]):
            compile_executor.add_include_path(parent)
        kwargs["inputs_from"] = [
            Phase(
                project=project,
                name="Render C",
                inputs=inputs,
                executor=RenderTemplateExecutor(output_extension="c").with_args(
                    template_args
                ),
            )
        ]
    else:
        kwargs["inputs"] = inputs
    return Phase(**kwargs)


def clang_link(project: Project, inputs_from: list[Phase], output: str):
    linker = GccLink("clang-12")
    linker.add_argument("-rpath", pathlib.Path.home() / ".local" / "lib")
    linker.optimize(3)
    Phase(
        project=project,
        name="Link Executable",
        inputs=[str(MLIR_RUNNER_UTILS), str(LAGRAD_UTILS)],
        inputs_from=inputs_from,
        executor=linker,
        output=output,
    )


def clang_dynamiclib(project: Project, inputs_from: list[Phase], output: str):
    build = GccLink()
    build.output_type = "library"
    build.output_extension = "dylib"
    build.add_argument("-dynamiclib")
    Phase(
        project=project,
        name="Build dynamic lib",
        inputs_from=inputs_from,
        executor=build,
        output=output,
    )