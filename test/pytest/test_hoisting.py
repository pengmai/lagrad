import os.path as osp
import re

MLIR_FILES = osp.join(osp.dirname(__file__), "..", "Standalone")

def test_single_loop():
    with open(f"{MLIR_FILES}/hoisting/single_loop.mlir") as f:
        contents = f.read()
    # This is atrocious. Ideally I'd be using the MLIR Python bindings to verify
    # transformations.
    m = re.match(r"(?:(?!scf).|\s)+scf\.for[^{]+{([^}]+)}", contents)
    print(m.group(1))
