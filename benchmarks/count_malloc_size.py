"""Need to determine how many bytes of memory the Enzyme program is allocating."""
import argparse
import re

pat = re.compile(r".*@malloc\(i64 (\d+)\).*")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()

    alloc_size = 0
    with open(args.input_file) as f:
        for line in f:
            match = pat.match(line)
            if match:
                alloc_size += int(match.group(1))
            elif "@malloc" in line:
                print(line)
    print(f"Total alloc size: {(alloc_size / 1e9)} GB")
