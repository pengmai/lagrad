"""
A utility script to quickly run Enzyme on a file. Don't know if I need this tbh.
"""

import argparse
import sys
from compile import compile_mlir_to_enzyme

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file) as f:
            contents = f.read().encode("utf-8")
    else:
        contents = sys.stdin.read().encode("utf-8")
    print(compile_mlir_to_enzyme(contents, emit="llvm"))
