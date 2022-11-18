import ctypes
import pathlib
import numpy as np
from numpy.typing import NDArray
import subprocess

double_ptr = ctypes.POINTER(ctypes.c_double)
float_ptr = ctypes.POINTER(ctypes.c_float)
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


class F32Descriptor1D(MemRefDescriptor):
    _fields_ = [
        ("allocated", float_ptr),
        ("aligned", float_ptr),
        ("offset", ctypes.c_longlong),
        ("size", ctypes.c_longlong),
        ("stride", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size]


class F32Descriptor2D(MemRefDescriptor):
    _fields_ = [
        ("allocated", float_ptr),
        ("aligned", float_ptr),
        ("offset", ctypes.c_longlong),
        ("size_0", ctypes.c_longlong),
        ("size_1", ctypes.c_longlong),
        ("stride_0", ctypes.c_longlong),
        ("stride_1", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size_0, self.size_1]


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


memref_1d_f32 = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 3
memref_2d_f32 = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 5
memref_1d = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 3
memref_1d_int = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 3
memref_1d_index = [
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 3
memref_2d = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
] + [ctypes.c_longlong] * 5
memref_2d_int = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"),
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
        assert isinstance(arr, np.float64) or isinstance(
            arr, np.int64
        ), f"Unexpected argument type: '{type(arr)}'"
        return (arr,)
    return (
        (arr, arr, 0)
        + arr.shape
        + tuple(stride // arr.itemsize for stride in arr.strides)
    )


TMP_DIR = pathlib.Path(__file__).parent / "build" / "osx64"
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


# compile_bindings()


class GMMFullGrad(ctypes.Structure):
    _fields_ = [
        ("dalphas", F64Descriptor1D),
        ("dmeans", F64Descriptor2D),
        ("dQs", F64Descriptor2D),
        ("dLs", F64Descriptor3D),
    ]


class BAReprojGrad(ctypes.Structure):
    _fields_ = [
        ("dcam", F64Descriptor1D),
        ("dX", F64Descriptor1D),
        ("dw", ctypes.c_double),
    ]


class HandComplicatedGrad(ctypes.Structure):
    _fields_ = [("dtheta", F64Descriptor1D), ("dus", F64Descriptor2D)]


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


class LSTMObjectiveGrad(ctypes.Structure):
    _fields_ = [
        ("dmain_params", F64Descriptor4D),
        ("dextra_params", F64Descriptor2D),
    ]


class NNGrad(ctypes.Structure):
    _fields_ = [
        ("dweight0", F32Descriptor2D),
        ("dbias0", F32Descriptor1D),
        ("dweight1", F32Descriptor2D),
        ("dbias1", F32Descriptor1D),
        ("dweight2", F32Descriptor2D),
        ("dbias2", F32Descriptor1D),
    ]


def struct_to_tuple(s):
    if isinstance(s, float):
        return s
    elif isinstance(s, MemRefDescriptor):
        return s.to_numpy()
    descriptors = (getattr(s, field[0]) for field in s._fields_)
    return (
        (desc if isinstance(desc, float) else desc.to_numpy()) for desc in descriptors
    )


ctypes.CDLL(
    pathlib.Path.home() / ".local" / "LLVM16" / "lib" / "libmlir_runner_utils.dylib"
)
ctypes.CDLL(
    pathlib.Path.home() / ".local" / "LLVM16" / "lib" / "libmlir_c_runner_utils.dylib"
)
mlirlib = ctypes.CDLL(TMP_DIR / "mlir_bindings.dylib")
gmm_args = (
    memref_1d
    + memref_2d
    + memref_2d
    + memref_3d
    + memref_2d
    + [ctypes.c_double, ctypes.c_longlong]
)
mlirlib.gmm_objective_full.argtypes = gmm_args
mlirlib.gmm_objective_full.restype = ctypes.c_double
mlirlib.lagrad_gmm_full.argtypes = gmm_args
mlirlib.lagrad_gmm_full.restype = GMMFullGrad

mlirlib.lagrad_compute_reproj_error.argtypes = (
    memref_1d
    + memref_1d
    + [ctypes.c_double]
    + memref_1d
    + memref_1d
    + [ctypes.c_longlong]
)
mlirlib.lagrad_compute_reproj_error.restype = BAReprojGrad
mlirlib.lagrad_compute_w_error.argtypes = [ctypes.c_double]
mlirlib.lagrad_compute_w_error.restype = ctypes.c_double

DISABLE_HAND = False
if not DISABLE_HAND:
    mlirlib.mto_pose_params.argtypes = memref_1d
    mlirlib.mto_pose_params.restype = F64Descriptor2D
    mlirlib.lagrad_to_pose_params.argtypes = memref_1d
    mlirlib.lagrad_to_pose_params.restype = F64Descriptor1D
    mlirlib.mget_posed_relatives.argtypes = memref_3d + memref_2d
    mlirlib.mget_posed_relatives.restype = F64Descriptor3D
    mlirlib.lagrad_get_posed_relatives.argtypes = memref_3d + memref_2d
    mlirlib.lagrad_get_posed_relatives.restype = F64Descriptor2D
    mlirlib.mrelatives_to_absolutes.argtypes = memref_3d + memref_1d_int
    mlirlib.mrelatives_to_absolutes.restype = F64Descriptor3D
    mlirlib.lagrad_relatives_to_absolutes.argtypes = memref_3d + memref_1d_int
    mlirlib.lagrad_relatives_to_absolutes.restype = F64Descriptor3D
    mlirlib.HELPER_get_transforms.argtypes = (
        memref_1d + memref_1d_int + memref_3d + memref_3d
    )
    mlirlib.HELPER_get_transforms.restype = F64Descriptor3D
    mlirlib.lagrad_skinned_vertex_subset.argtypes = memref_3d + memref_2d + memref_2d
    mlirlib.lagrad_skinned_vertex_subset.restype = F64Descriptor3D
    hand_objective_args = (
        memref_1d
        + memref_1d_int
        + memref_3d
        + memref_3d
        + memref_2d
        + memref_2d
        + memref_1d_int
        + memref_2d
    )
    mlirlib.mlir_hand_objective.argtypes = hand_objective_args
    mlirlib.mlir_hand_objective.restype = F64Descriptor2D
    mlirlib.lagrad_hand_objective.argtypes = hand_objective_args + memref_2d
    mlirlib.lagrad_hand_objective.restype = F64Descriptor1D
    hand_complicated_args = (
        memref_1d
        + memref_2d
        + memref_1d_int
        + memref_3d
        + memref_3d
        + memref_2d
        + memref_2d
        + memref_2d_int
        + memref_1d_int
        + memref_2d
    )
    mlirlib.mlir_hand_objective_complicated.argtypes = hand_complicated_args
    mlirlib.mlir_hand_objective_complicated.restype = F64Descriptor2D
    mlirlib.lagrad_hand_objective_complicated.argtypes = (
        hand_complicated_args + memref_2d
    )
    mlirlib.lagrad_hand_objective_complicated.restype = HandComplicatedGrad

DISABLE_LSTM = False
if not DISABLE_LSTM:
    mlirlib.lagrad_lstm_model.argtypes = (
        memref_2d + memref_2d + memref_1d + memref_1d + memref_1d
    )
    mlirlib.lagrad_lstm_model.restype = LSTMModelGrad
    mlirlib.lagrad_lstm_predict.argtypes = memref_4d + memref_2d + memref_3d + memref_1d
    mlirlib.lagrad_lstm_predict.restype = LSTMPredictGrad
    mlirlib.lagrad_lstm_objective.argtypes = (
        memref_4d + memref_2d + memref_3d + memref_2d
    )
    mlirlib.lagrad_lstm_objective.restype = LSTMObjectiveGrad
mlp_args = (
    memref_2d_f32
    + memref_1d_int
    + memref_2d_f32
    + memref_1d_f32
    + memref_2d_f32
    + memref_1d_f32
    + memref_2d_f32
    + memref_1d_f32
)
mlirlib.mlir_mlp_batched.argtypes = mlp_args
mlirlib.mlir_mlp_batched.restype = ctypes.c_float
mlirlib.lagrad_mlp_batched.argtypes = mlp_args
mlirlib.lagrad_mlp_batched.restype = NNGrad

# mlirlib.onehot_adjoint_err_nest.argtypes = memref_2d + memref_1d_index + memref_1d_int
# mlirlib.onehot_adjoint_err_nest.restype = F64Descriptor2D
# mlirlib.rowhot_insert.argtypes = memref_2d + [ctypes.c_longlong]
# mlirlib.rowhot_insert.restype = F64Descriptor2D
# mlirlib.onehot_square.argtypes = memref_2d + memref_1d_index
# mlirlib.onehot_square.restype = F64Descriptor2D
# mlirlib.onehot_sumreduce.argtypes = memref_2d + memref_1d_index
# mlirlib.onehot_sumreduce.restype = F64Descriptor1D
# mlirlib.onehot_matmul_both_transposed.argtypes = memref_2d + memref_2d + memref_1d_index
# mlirlib.onehot_matmul_both_transposed.restype = F64Descriptor2D
# mlirlib.onehot_matmul.argtypes = memref_2d + memref_2d + memref_1d_index
# mlirlib.onehot_matmul.restype = F64Descriptor2D
# mlirlib.rowhot_broadcast_mul.argtypes = memref_1d + memref_2d + [ctypes.c_longlong]
# mlirlib.rowhot_broadcast_mul.restype = F64Descriptor2D
# mlirlib.rowhot_matmul.argtypes = memref_2d + memref_2d + [ctypes.c_longlong]
# mlirlib.rowhot_matmul.restype = F64Descriptor2D
# mlirlib.colhot_broadcast_mul.argtypes = memref_1d + memref_2d + [ctypes.c_longlong]
# mlirlib.colhot_broadcast_mul.restype = F64Descriptor2D


def wrap(mlir_func):
    def wrapped(*args):
        args = tuple(arg for ndarr in args for arg in ndto_args(ndarr))
        return struct_to_tuple(mlir_func(*args))

    return wrapped


mlir_gmm_primal_full = wrap(mlirlib.gmm_objective_full)
lagrad_gmm_full = wrap(mlirlib.lagrad_gmm_full)
lagrad_ba_compute_reproj_error = wrap(mlirlib.lagrad_compute_reproj_error)
lagrad_ba_compute_w_error = wrap(mlirlib.lagrad_compute_w_error)


def notimplemented(*_):
    raise NotImplementedError()


if DISABLE_HAND:
    hand_to_pose_params = notimplemented
    lagrad_hand_to_pose_params = notimplemented
    hand_get_posed_relatives = notimplemented
    lagrad_get_posed_relatives = notimplemented
    hand_relatives_to_absolutes = notimplemented
    lagrad_relatives_to_absolutes = notimplemented
    mlir_HELPER_get_transforms = notimplemented
    lagrad_skinned_vertex_subset = notimplemented
    mlir_hand_objective = notimplemented
    mlir_hand_objective_complicated = notimplemented
    lagrad_hand_objective = notimplemented
    lagrad_hand_objective_complicated = notimplemented
else:
    hand_to_pose_params = wrap(mlirlib.mto_pose_params)
    lagrad_hand_to_pose_params = wrap(mlirlib.lagrad_to_pose_params)
    hand_get_posed_relatives = wrap(mlirlib.mget_posed_relatives)
    lagrad_get_posed_relatives = wrap(mlirlib.lagrad_get_posed_relatives)
    hand_relatives_to_absolutes = wrap(mlirlib.mrelatives_to_absolutes)
    lagrad_relatives_to_absolutes = wrap(mlirlib.lagrad_relatives_to_absolutes)
    mlir_HELPER_get_transforms = wrap(mlirlib.HELPER_get_transforms)
    lagrad_skinned_vertex_subset = wrap(mlirlib.lagrad_skinned_vertex_subset)
    mlir_hand_objective = wrap(mlirlib.mlir_hand_objective)
    mlir_hand_objective_complicated = wrap(mlirlib.mlir_hand_objective_complicated)
    lagrad_hand_objective = wrap(mlirlib.lagrad_hand_objective)
    lagrad_hand_objective_complicated = wrap(mlirlib.lagrad_hand_objective_complicated)

if DISABLE_LSTM:
    lagrad_lstm_model = notimplemented
    lagrad_lstm_predict = notimplemented
    lagrad_lstm_objective = notimplemented
else:
    lagrad_lstm_model = wrap(mlirlib.lagrad_lstm_model)
    lagrad_lstm_predict = wrap(mlirlib.lagrad_lstm_predict)
    lagrad_lstm_objective = wrap(mlirlib.lagrad_lstm_objective)

mlir_mlp_primal = wrap(mlirlib.mlir_mlp_batched)
lagrad_mlp = wrap(mlirlib.lagrad_mlp_batched)

# onehot_adjoint_err_nest = wrap(mlirlib.onehot_adjoint_err_nest)
# rowhot_insert = wrap(mlirlib.rowhot_insert)
# onehot_square = wrap(mlirlib.onehot_square)
# onehot_sumreduce = wrap(mlirlib.onehot_sumreduce)
# onehot_matmul_both_transposed = wrap(mlirlib.onehot_matmul_both_transposed)
# onehot_matmul = wrap(mlirlib.onehot_matmul)
# rowhot_broadcast_mul = wrap(mlirlib.rowhot_broadcast_mul)
# rowhot_matmul = wrap(mlirlib.rowhot_matmul)
# colhot_broadcast_mul = wrap(mlirlib.colhot_broadcast_mul)
