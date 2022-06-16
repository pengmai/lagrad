import ctypes
import pathlib
import numpy as np
from numpy.typing import NDArray
import subprocess

double_ptr = ctypes.POINTER(ctypes.c_double)
cstdlib = ctypes.cdll.LoadLibrary("libSystem.dylib")
cstdlib.free.argtypes = [ctypes.c_void_p]
cstdlib.free.restype = ctypes.c_void_p


class MemRefDescriptor(ctypes.Structure):
    freed = False
    nparr: NDArray = None

    def to_numpy(self):
        assert not self.freed, "Memory was freed"
        if not self.nparr:
            self.nparr = np.ctypeslib.as_array(self.aligned, self.shape).copy()
        self.free()
        return self.nparr

    def free(self):
        assert not self.freed, "Memory was already freed"
        cstdlib.free(self.allocated)
        self.freed = True


class F64Descriptor1D(MemRefDescriptor):
    _fields_ = [
        ("allocated", double_ptr),
        ("aligned", double_ptr),
        ("offset", ctypes.c_longlong),
        ("size", ctypes.c_longlong),
        ("stride", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size]


class F64Descriptor2D(MemRefDescriptor):
    _fields_ = [
        ("allocated", double_ptr),
        ("aligned", double_ptr),
        ("offset", ctypes.c_longlong),
        ("size_0", ctypes.c_longlong),
        ("size_1", ctypes.c_longlong),
        ("stride_0", ctypes.c_longlong),
        ("stride_1", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size_0, self.size_1]


class F64Descriptor3D(MemRefDescriptor):
    _fields_ = [
        ("allocated", double_ptr),
        ("aligned", double_ptr),
        ("offset", ctypes.c_longlong),
        ("size_0", ctypes.c_longlong),
        ("size_1", ctypes.c_longlong),
        ("size_2", ctypes.c_longlong),
        ("stride_0", ctypes.c_longlong),
        ("stride_1", ctypes.c_longlong),
        ("stride_2", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size_0, self.size_1, self.size_2]


class F64Descriptor4D(MemRefDescriptor):
    _fields_ = [
        ("allocated", double_ptr),
        ("aligned", double_ptr),
        ("offset", ctypes.c_longlong),
        ("size_0", ctypes.c_longlong),
        ("size_1", ctypes.c_longlong),
        ("size_2", ctypes.c_longlong),
        ("size_3", ctypes.c_longlong),
        ("stride_0", ctypes.c_longlong),
        ("stride_1", ctypes.c_longlong),
        ("stride_2", ctypes.c_longlong),
        ("stride_3", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size_0, self.size_1, self.size_2, self.size_3]


memref_1d = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 3
memref_2d = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 5
memref_3d = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 7
memref_4d = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=4, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=4, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 9


def ndto_args(arr):
    if not isinstance(arr, np.ndarray):
        assert isinstance(arr, np.float64), f"Unexpected argument type: '{type(arr)}'"
        return (arr,)
    return (
        (arr, arr, 0)
        + arr.shape
        + tuple(stride // arr.itemsize for stride in arr.strides)
    )


TMP_DIR = pathlib.Path(__file__).parent / "tmp"
BENCHMARK_TEMPLATES = (
    pathlib.Path(__file__).parents[2] / "benchmarks" / "mlir" / "templates"
)
HAND_MLIR_FILE = f"{BENCHMARK_TEMPLATES}/hand.mlir"


def compile_bindings(verbose=False):
    try:
        opt_p = subprocess.run(
            ["make", "-C", pathlib.Path(__file__).parent],
            capture_output=True,
            check=True,
        )
        if verbose:
            print(opt_p.stdout.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr.decode("utf-8"))


compile_bindings()


class BAReprojGrad(ctypes.Structure):
    _fields_ = [
        ("dcam", F64Descriptor1D),
        ("dX", F64Descriptor1D),
        ("dw", ctypes.c_double),
    ]


class LSTMModelGrad(ctypes.Structure):
    _fields_ = [
        ("dweight", F64Descriptor2D),
        ("dbias", F64Descriptor2D),
        ("dhidden", F64Descriptor1D),
        ("dcell", F64Descriptor1D),
        ("dinput", F64Descriptor1D),
    ]


class LSTMPredictGrad(ctypes.Structure):
    _fields_ = [
        ("dmain_params", F64Descriptor4D),
        ("dextra_params", F64Descriptor2D),
        ("dstate", F64Descriptor3D),
    ]


def struct_to_tuple(s):
    if isinstance(s, float):
        return s
    descriptors = (getattr(s, field[0]) for field in s._fields_)
    return (
        (desc if isinstance(desc, float) else desc.to_numpy()) for desc in descriptors
    )


mlirlib = ctypes.CDLL(TMP_DIR / "mlir_bindings.dylib")
mlirlib.lagrad_compute_reproj_error.argtypes = (
    memref_1d + memref_1d + [ctypes.c_double] + memref_1d + memref_1d
)
mlirlib.lagrad_compute_reproj_error.restype = BAReprojGrad
mlirlib.lagrad_compute_w_error.argtypes = [ctypes.c_double]
mlirlib.lagrad_compute_w_error.restype = ctypes.c_double

mlirlib.lagrad_lstm_model.argtypes = (
    memref_2d + memref_2d + memref_1d + memref_1d + memref_1d
)
mlirlib.lagrad_lstm_model.restype = LSTMModelGrad
mlirlib.lagrad_lstm_predict.argtypes = memref_4d + memref_2d + memref_3d + memref_1d
mlirlib.lagrad_lstm_predict.restype = LSTMPredictGrad


def wrap(mlir_func):
    def wrapped(*args):
        args = tuple(arg for ndarr in args for arg in ndto_args(ndarr))
        return struct_to_tuple(mlir_func(*args))

    return wrapped


lagrad_ba_compute_reproj_error = wrap(mlirlib.lagrad_compute_reproj_error)
lagrad_ba_compute_w_error = wrap(mlirlib.lagrad_compute_w_error)
lagrad_lstm_model = wrap(mlirlib.lagrad_lstm_model)
lagrad_lstm_predict = wrap(mlirlib.lagrad_lstm_predict)
