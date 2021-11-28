import os.path as osp
from toolchain import jit_file
from stdout_parser import extract_scalar, extract_1d, extract_2d, extract_3d

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")


def test_pose_relatives_body():
    expected = [40.8654, 29.4633, -25.8706]
    assert (
        extract_1d(jit_file(f"{MLIR_FILES}/hand/get_posed_relatives_body.mlir"))
        == expected
    )
