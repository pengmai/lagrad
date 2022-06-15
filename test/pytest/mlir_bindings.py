import ctypes
import numpy as np
from numpy.typing import NDArray

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
            self.nparr = np.ctypeslib.as_array(self.aligned, self.shape)
        return self.nparr

    def free(self):
        assert not self.freed, "Memory was already freed"
        cstdlib.free(self.allocated)
        self.freed = True

    def __enter__(self):
        return self.to_numpy()

    def __exit__(self, typ, value, traceback):
        if not self.freed:
            self.free()


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
    return (
        (arr, arr, 0)
        + arr.shape
        + tuple(stride // arr.itemsize for stride in arr.strides)
    )
