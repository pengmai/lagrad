#!/usr/bin/env python

import argparse
from jinja2 import Template

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--output", "-o")

    args, unknown = parser.parse_known_args()

    def get_value(v: str):
        return int(v) if v.isnumeric() else v

    template_args = {k[1:]: get_value(v) for k, v in zip(unknown[::2], unknown[1::2])}

    with open(args.input_file, "r") as f:
        template = Template(f.read())
    rendered = template.render(**template_args)
    if args.output:
        with open(args.output, "w") as f:
            f.write(rendered)
    else:
        print(rendered)
