import re
import json


def remove_whitespace(output: str):
    return re.sub(r"\s+", "", output)


float_regex = r"[-+]?[0-9]*\.?[0-9]+"


def extract_scalar(output: str):
    # Remove all newlines
    output = remove_whitespace(output)
    pat = re.compile(rf"data=\[({float_regex})\]")
    m = pat.search(output)
    return float(m.group(1))


def extract_1d(output: str):
    output = remove_whitespace(output)
    pat = re.compile(rf"data=\[({float_regex}(?:,{float_regex})*)\]")
    m = pat.search(output).group(1)
    return [float(el) for el in m.split(",")]


def extract_2d(output: str):
    output = remove_whitespace(output)
    arr_regex = rf"\[{float_regex}(?:,{float_regex})*\]"
    mat_regex = re.compile(rf"data=(\[{arr_regex}(?:,{arr_regex})*\])")
    m = json.loads(mat_regex.search(output).group(1))
    return m
