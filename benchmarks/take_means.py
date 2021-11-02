import argparse
import json
import pandas as pd


def main(args):
    with open(args.results_file) as f:
        contents = json.load(f)
    print(contents["config"])
    results = pd.DataFrame.from_dict(contents["results"])
    print(results[args.warmups :].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", "-r", required=True)
    parser.add_argument("--warmups", "-w", type=int, default=10)
    args = parser.parse_args()

    main(args)
