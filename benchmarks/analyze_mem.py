import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"


def ba_main():
    filepat = re.compile(r"ba\d+_n(?P<n>\d+)_m(?P<m>\d+)_p(?P<p>\d+).txt")
    df = pd.read_csv(
        "detailed_results/ba_memusage.csv", header=[0, 1, 2], index_col=0, skiprows=1
    )
    # There's a memory leak that causes the consumption to go up with each run, so we just take the first run.
    df = df[:1].unstack()
    df = df[:, :, "rss", "run1"] / 1e6
    sizes = df.index.get_level_values(0).str.extract(filepat).apply(pd.to_numeric)
    sizes["jacobian_cols"] = 11 * sizes.n + 3 * sizes.m + sizes.p
    idx = pd.MultiIndex.from_arrays(
        [sizes["jacobian_cols"], df.index.get_level_values(1)]
    )
    df.index = idx
    df.sort_index(level=[0, 1], inplace=True)
    print(df.index)

    fig, ax = plt.subplots(1, 1)
    for key in ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]:
        series = df[:, key]
        ax.plot(series.index, series, label=key, marker=".")
    ax.legend()
    ax.set_title("Memory consumption of BA")
    ax.set_xlabel("Number of columns in Jacobian")
    ax.set_ylabel("Peak resident set size (MB)")
    ax.set_yscale("log")

    plt.show()


def main_term_main():
    mem_df = pd.read_csv(
        f"detailed_results/main_term_memusage.tsv",
        sep="\t",
        index_col=[0, 1],
        # skiprows=1,
    )
    # Ignore vsize for now.
    filepat = re.compile(r"gmm_d(?P<d>\d+)_K(?P<k>\d+).txt")
    sizes = mem_df.index.get_level_values(0).str.extract(filepat).apply(pd.to_numeric)
    sizes["jacobian"] = sizes.k + sizes.d * sizes.k * 2 + sizes.k * sizes.d ** 2
    idx = pd.MultiIndex.from_arrays(
        [sizes["jacobian"], mem_df.index.get_level_values(1)]
    )
    mem_df.index = idx

    fig, ax = plt.subplots(1, 1)
    for key in ["LAGrad", "Enzyme/C", "Handwritten"]:
        series = mem_df["max_rss"][:, key] / 1e6
        ax.plot(series.index, series, label=key, marker=".")
    ax.legend()
    ax.set_title("Memory consumption of Main Term (GMM subset)")
    ax.set_xlabel("Number of independent variables")
    ax.set_ylabel("Peak resident set size (MB)")
    ax.set_yscale("log")
    plt.show()


if __name__ == "__main__":
    # main()
    ba_main()
