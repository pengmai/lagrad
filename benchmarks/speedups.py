import pandas as pd
import numpy as np
from matplotlib import rcParams
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# from scipy import stats
from speedups_data import *

sns.set()
rcParams["font.family"] = "Times New Roman"


def pad(lst: np.ndarray, length=5):
    return np.pad(lst, ((0, 0), (0, length - lst.shape[1])), "constant")


def process(lst: list):
    return pad(np.clip(np.array(lst), a_min=0.0, a_max=None), 5)


def set_margins():
    x_margin = 0.07
    plt.subplots_adjust(
        left=0 + x_margin,
        bottom=0 + 0.14,
        right=1 - x_margin,
        top=1 - 0.08,
        wspace=0,
        hspace=0,
    )


def plot_stacks():
    labels = pd.MultiIndex.from_product(
        (["trmv", "gmm", "ba", "lstm", "hand rt"], ["small", "medium", "large"])
    )
    gmm_data = [
        [0.45, -0.01, -0.17, 0.53],
        [0.15, 0.00, 1.67, 1.08],
        [0.53, 0.00, 0.29, 2.07],
    ]
    # size x opt level
    trmv_data = [
        [0.00] * 2 + [0.86, 0.02],
        [0.00] * 2 + [1.11, 0.22],
        [0.00] * 2 + [1.06, 0.08],
    ]
    ba_data = [[0.23, 7.28], [0.25, 7.59], [0.26, 7.66]]
    lstm_data = [[0.01, 0.01], [0.05, 0.00], [0.01, 0.02]]
    hand_rt_data = [
        [0.03, 0.00, 0.00, 0.00, 0.76],
        [0.02, 0.02, 0.00, 0.00, 0.76],
        [0.00, 0.00, 0.00, 0.00, 1.23],
    ]
    trmv_data = process(trmv_data)
    gmm_data = process(gmm_data)
    ba_data = process(ba_data)
    lstm_data = process(lstm_data)
    hand_rt_data = process(hand_rt_data)
    all_data = np.vstack((trmv_data, gmm_data, ba_data, lstm_data, hand_rt_data)).T
    # all_data = np.pad(all_data, ((1, 0), (0, 0)), "constant", constant_values=1.0)
    # print(all_data)

    fig, ax = plt.subplots()
    opt_labels = [
        "In-place Bufferization",
        "Stack Buffer Promotion",
        "Tri Comp",
        "Tri Packing",
        "Adjoint Sparsity",
    ]
    bottoms = np.ones_like(all_data[0])
    x = np.arange(len(labels))
    width = 0.5
    for i, row in enumerate(all_data):
        rects = ax.bar(x, row, bottom=bottoms, label=opt_labels[i], width=width)
        # ax.bar_label(rects, padding=3)
        bottoms = bottoms + row
    # ax.bar(x, np.ones_like(all_data[0]), label="Baseline", width=width)
    ax.set_ylabel("Speedup")
    ax.set_yscale("log")
    yticks = 2 ** (np.arange(5).astype(float))
    ax.set_yticks(yticks, yticks)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    set_hierarchical_xlabels(x, labels, ax)
    ax.legend()
    fig.tight_layout()
    set_margins()
    plt.show()


def set_hierarchical_xlabels(
    x,
    index,
    ax=None,
    bar_xmargin=0.1,  # Margins on the left and right ends of the line, X-axis scale
    bar_yinterval=0.08,  # Relative value with the vertical spacing of the line and the length of the Y axis as 1?
):
    from itertools import groupby
    from matplotlib.lines import Line2D

    ax = ax or plt.gca()

    assert isinstance(index, pd.MultiIndex)
    ax.set_xticks(x, [s for *_, s in index])

    transform = ax.get_xaxis_transform()

    for i in range(1, len(index.codes)):
        xpos0 = -0.5  # Coordinates on the left side of the target group
        for (*_, code), codes_iter in groupby(zip(*index.codes[:-i])):
            xpos1 = xpos0 + sum(
                1 for _ in codes_iter
            )  # Coordinates on the right side of the target group
            ax.text(
                (xpos0 + xpos1) / 2,
                (bar_yinterval * (-i - 0.1)),
                index.levels[-i - 1][code],
                transform=transform,
                ha="center",
                va="top",
            )
            ax.add_line(
                Line2D(
                    [xpos0 + bar_xmargin, xpos1 - bar_xmargin],
                    [bar_yinterval * -i] * 2,
                    linewidth=0.5,
                    transform=transform,
                    color="k",
                    clip_on=False,
                )
            )
            xpos0 = xpos1


def plot_vs_sota():
    trmv_data = get_trmv_data()
    gmm_data = get_gmm_data()
    lstm_data = get_lstm_data()
    hand_data = get_hand_data()
    ba_data = get_ba_data()
    mlp_data = get_mlp_data()

    labels = pd.MultiIndex.from_product(
        (
            ["trmv", "gmm", "ba", "lstm", "hand rt", "mnist mlp"],
            ["small", "medium", "large"],
        )
    )
    results = np.hstack((trmv_data, gmm_data, ba_data, lstm_data, hand_data, mlp_data))
    print(results)
    # print(results)

    fig, ax = plt.subplots()
    width = 0.1

    powers = 2 ** np.arange(8) # default = 4
    powers = np.concatenate(((1 / powers)[1:][::-1], powers))
    methods = ["LAGrad", "Enzyme/MLIR", "Enzyme/C", "PyTorch"]
    x = np.arange(len(labels))
    ones = np.ones_like(results[0])
    i_off = 0
    # ax.add_line(matplotlib.lines.Line2D([-1, len(labels)], [1.0] * 2, linewidth=0.5))
    for i, (method, result) in enumerate(zip(methods, results)):
        if i == 1:
            continue
        offset_x = x + width * (i_off - 1)
        bplot = ax.bar(offset_x, result - ones, width, bottom=ones, label=method)
        for point, maybenan in zip(offset_x, result):
            if np.isnan(maybenan):
                ax.plot(point, 1.0, "x", color="red")
            elif maybenan < powers[0]:
                s = f"{maybenan:.1e}"
                m, e = s.split("e")
                ax.text(
                    point - (0.2 if i_off == 1 else -0.1),
                    powers[0] + 0.01,
                    f"${m:s}\\times 10^{{{int(e):d}}}$",
                    math_fontfamily="stix",
                    size="small",
                    horizontalalignment="left",
                    rotation="vertical",
                )

        i_off += 1

    ax.legend()

    ax.set_ylabel("Speedup")
    ax.set_yscale("log")
    ax.set_yticks(powers, powers)
    ax.grid(axis="x")
    ax.set_ylim(powers[0], powers[-1])
    set_hierarchical_xlabels(x, labels, ax, bar_yinterval=0.08)
    fig.tight_layout()
    set_margins()
    plt.show()


def plot_mem_vs():
    results = np.hstack(
        (trmv_memory(), gmm_memory(), ba_memory(), lstm_memory(), hand_memory())
    )
    print(results)
    labels = pd.MultiIndex.from_product(
        (["trmv", "gmm", "ba", "lstm", "hand rt"], ["small", "medium", "large"])
    )

    fig, ax = plt.subplots()
    width = 0.1

    exponents = (-(np.arange(8) - 1))[::-1]
    powers = 2 ** exponents.astype(float)
    # powers = powers[::-1]
    methods = ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]
    x = np.arange(len(labels))
    for i, (method, result) in enumerate(zip(methods, results)):
        offset_x = x + width * (i - 1)
        ax.bar(offset_x, result, width, label=method)
        # for point, maybenan in zip(offset_x, result):
        #     if np.isnan(maybenan):
        #         ax.plot(point, powers[0] + 3e-3, "x", color="red")

    ax.legend()

    ax.set_ylabel("Relative Memory Consumption Reduction")
    ax.set_yscale("log")
    ax.set_yticks(powers, [f"$2^{{{x}}}$" for x in exponents])
    ax.set_ylim(powers[0], powers[-1])
    set_hierarchical_xlabels(x, labels, ax, bar_yinterval=0.08)
    fig.tight_layout()
    set_margins()
    plt.show()


if __name__ == "__main__":
    # plot_stacks()
    # plot_mem_vs()
    plot_vs_sota()
