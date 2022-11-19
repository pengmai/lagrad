#!/usr/bin/env python

import re
import argparse


def replace_hand_optimization(lines: bytes, complicated: bool):
    memset_pattern = "@.memset_pattern = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 1.000000e+00, double 1.000000e+00], align 16"
    memset_call = (
        lambda ssa_val: f"  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(24) {ssa_val}, i8* bitcast ([3 x double]* @.memset_pattern to i8*), i64 24, i1 false)"
    )
    call_pat = re.compile(r"\s+call void @memset_pattern16\(i8\* (%\d+)")
    # if complicated:
    #     memset_call = "  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(24) %77, i8* bitcast ([3 x double]* @.memset_pattern to i8*), i64 24, i1 false)"

    def process_line(line: str):
        if line.lstrip().startswith("call void @memset_pattern16"):
            ssa_val = call_pat.match(line).group(1)
            return memset_call(ssa_val)
        elif line.startswith("@.memset_pattern.1 ="):
            return memset_pattern
        return line

    return "\n".join(
        [process_line(line) for line in lines.decode("utf-8").splitlines()]
    ).encode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--output", "-o")
    parser.add_argument("--complicated", action="store_true")
    parser.add_argument("--mode", required=True, choices=["mlir", "c"])

    args = parser.parse_args()

    with open(args.input_file, "rb") as f:
        contents = replace_hand_optimization(f.read(), args.complicated)
    with open(args.output, "wb") as f:
        f.write(contents)
