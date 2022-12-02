import pandas as pd
import numpy as np
from matplotlib import rcParams
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from scipy import stats
from speedups_data import *

sns.set()
rcParams["font.family"] = "Times New Roman"
# rcParams["figure.dpi"] = 400
# rcParams["figure.dpi"] = 100
dims = (12, 3.5)
# dims = (8, 3)


def pad(lst: np.ndarray, length=5):
    return np.pad(lst, ((0, 0), (0, length - lst.shape[1])), "constant")


def process(lst: list):
    return pad(np.clip(np.array(lst), a_min=0.0, a_max=None), 6)


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


def plot_stacks(save=False):
    labels = pd.MultiIndex.from_product(
        (
            ["trmv", "gmm", "ba", "lstm", "hand", "mnist mlp"],
            ["small", "medium", "large"],
        )
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
    mlp_data = [
        [0.00, 0.13, 0, 0, 0, 45.81],
        [0.00, 0.05, 0, 0, 0, 49.99],
        [0, 0, 0, 0, 0, 64.14],
    ]
    trmv_data = process(trmv_data)
    gmm_data = process(gmm_data)
    ba_data = process(ba_data)
    lstm_data = process(lstm_data)
    hand_rt_data = process(hand_rt_data)
    mlp_data = process(mlp_data)
    all_data = np.vstack(
        (trmv_data, gmm_data, ba_data, lstm_data, hand_rt_data, mlp_data)
    ).T
    # all_data = np.pad(all_data, ((1, 0), (0, 0)), "constant", constant_values=1.0)
    # print(all_data)

    fig, ax = plt.subplots(figsize=dims)
    opt_labels = [
        "In-place Bufferization",
        "Stack Buffer Promotion",
        "Tri Comp",
        "Tri Packing",
        "Adjoint Sparsity",
        "OpenBLAS Integration",
    ]
    bottoms = np.ones_like(all_data[0])
    x = np.arange(len(labels))
    yticks = 2 ** (np.arange(5).astype(float))
    width = 0.4
    for i, row in enumerate(all_data):
        rects = ax.bar(x, row, bottom=bottoms, label=opt_labels[i], width=width)
        for i, (value, bottom) in enumerate(zip(row, bottoms)):
            v = value + bottom
            if v > yticks[-1]:
                ax.text(
                    i,
                    yticks[-1],
                    f"${v:.0f}\\times$",
                    math_fontfamily="stix",
                    size="small",
                    horizontalalignment="center",
                )
        # ax.bar_label(rects, padding=3)
        bottoms = bottoms + row
    # ax.bar(x, np.ones_like(all_data[0]), label="Baseline", width=width)
    ax.set_ylabel("Speedup")
    ax.set_yscale("log")
    ax.set_ylim(1, yticks[-1])
    ax.set_yticks(yticks, yticks)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    set_hierarchical_xlabels(x, labels, ax)
    ax.grid(axis="x")
    ax.legend()
    fig.tight_layout()
    set_margins()
    if save:
        plt.savefig("testfigure.pdf")
    else:
        plt.show()


def set_hierarchical_xlabels(
    x,
    index,
    ax=None,
    bar_xmargin=0.1,  # Margins on the left and right ends of the line, X-axis scale
    bar_yinterval=0.1,  # Relative value with the vertical spacing of the line and the length of the Y axis as 1?
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


def plot_vs_sota(save=False, action_text=False):
    print("Plotting speedup graphs vs Enzyme/PyTorch")
    trmv_data = get_trmv_data()
    gmm_data = get_gmm_data()
    lstm_data = get_lstm_data()
    hand_data = get_hand_data()
    ba_data = get_ba_data()
    mlp_data = get_mlp_data()

    labels = pd.MultiIndex.from_product(
        (
            ["trmv", "gmm", "ba", "lstm", "hand", "mnist mlp"],
            ["small", "medium", "large"],
        )
    )
    # labels = ["trmv", "gmm", "ba", "lstm", "hand", "mnist mlp"]
    results = np.hstack((trmv_data, gmm_data, ba_data, lstm_data, hand_data, mlp_data))
    results = np.delete(results, (1), axis=0)
    # results = results / results[-1]
    # first_results = results[:, 0::3]
    # results[]
    # results = results[:, 2::3]
    # results[:, 2] = first_results[:, 2]
    # results[:, 4] = first_results[:, 4]
    # print(max(results[0] / results[2]))
    print(results)

    fig, ax = plt.subplots(figsize=dims)
    width = 0.15

    powers = 2 ** np.arange(4)  # default = 4
    powers = np.concatenate(((1 / powers)[1:][::-1], powers))
    powers = np.concatenate((powers, 2 ** np.arange(4, 6)))
    print(powers)
    # powers = np.array([0.25,0.5, 1, 2, 4])
    methods = ["LAGrad", "Enzyme/C", "PyTorch"]
    # methods = ["LAGrad", "PyTorch"]
    x = np.arange(len(labels))
    ones = np.ones_like(results[0])
    ax.plot([-1, len(labels)], [1.0] * 2, color="black", label="Enzyme/MLIR")
    # colors = ["c", "m"]
    timeoutplotted = False
    for i, (method, result) in enumerate(zip(methods, results)):
        offset_x = x + width * (i - 1)
        # offset_x = x + width * (i - 0.5)
        bplot = ax.bar(offset_x, result - ones, width, bottom=ones, label=method)
        for point, maybenan in zip(offset_x, result):
            if np.isnan(maybenan):
                ax.plot(
                    point,
                    1.0,
                    "x",
                    color="red",
                    label="Timeout" if not timeoutplotted else None,
                )
                timeoutplotted = True
            elif maybenan < powers[0]:
                s = f"{maybenan:.1e}"
                m, e = s.split("e")
                ax.text(
                    point - (0.25 if i == 1 else -0.12),
                    # point - 0.25,
                    powers[0] + 0.01,
                    f"${m:s}\\times 10^{{{int(e):d}}}$",
                    math_fontfamily="stix",
                    size="small",
                    horizontalalignment="left",
                    rotation="vertical",
                )
            elif maybenan > powers[-1]:
                ax.text(
                    point - (0.1 if i == 0 else -0.15),
                    powers[-1],
                    f"${maybenan:.0f}\\times$",
                    math_fontfamily="stix",
                    size="small",
                    horizontalalignment="center",
                )

    ax.legend()

    # Impact text
    if action_text:
        for text, (tx, ty) in [
            ("8$\\times$ speedup\nfrom active\nsparsity", (1, 1 / 2)),
            # ("3.4$\\times$ speedup\nfrom adjoint\nsparsity", (13, 64)),
            ("100$\\times$ speedup\nfrom high\nperformance\nlibraries", (16, 1 / 2)),
        ]:
            ax.text(
                tx,
                ty,
                text,
                math_fontfamily="stix",
                size=30,
                color="crimson",
                horizontalalignment="center",
                verticalalignment="top",
            )

    # ax.set_title("Speedup of LAGrad vs Enzyme and PyTorch (Higher is Better)")
    ax.set_ylabel("Speedup")
    ax.set_yscale("log")
    ylabels = [(f"1/{int(1/x)}" if x < 1 else f"{int(x)}") for x in powers]
    # ylabels = [0.25, 1, 2, 3, 4]
    # ax.set_yticks(powers, powers)
    ax.set_yticks(powers, ylabels)
    ax.grid(axis="x")
    ax.set_ylim(powers[0], powers[-1])
    # ax.set_ylim(0.25, powers[-1])
    ax.set_xlim((-0.8, x[-1] + 0.8))
    # ax.set_xticks(x, labels)
    set_hierarchical_xlabels(x, labels, ax)
    fig.tight_layout()
    set_margins()
    if save:
        plt.savefig("testfigure.pdf")
    else:
        plt.show()


def plot_mem_vs(save=False, action_text=False):
    print("Plotting memory usage graphs")
    results = np.hstack(
        (
            trmv_memory(),
            gmm_memory(),
            ba_memory(),
            lstm_memory(),
            hand_memory(),
            nn_memory(),
        )
    )
    results = np.delete(results, (1), axis=0)
    # first_datasets = results[:, 0::3]
    # results = results[:, 2::3]  # Only take the large datasets
    # results[:, 2] = first_datasets[:, 2]
    # results[:, 4] = first_datasets[:, 4]

    # print(results)
    # return
    labels = pd.MultiIndex.from_product(
        (
            ["trmv", "gmm", "ba", "lstm", "hand", "mnist mlp"],
            ["small", "medium", "large"],
        )
    )
    # labels = ["trmv", "gmm", "ba", "lstm", "hand rt", "mnist mlp"]
    fig, ax = plt.subplots(figsize=dims)
    # width = 0.1
    width = 0.15

    pos_ticks = 6
    powers = 2 ** (np.arange(pos_ticks * 2 + 1).astype(float) - pos_ticks)
    powers = powers[3:]
    methods = ["LAGrad", "Enzyme/C", "PyTorch"]
    # methods = ["LAGrad", "PyTorch"]
    x = np.arange(len(labels))

    # ax.add_line(matplotlib.lines.Line2D([-1, len(labels)], [1.0] * 2, linewidth=0.9, color='black'), label="Enzyme/MLIR")
    ones = np.ones_like(results[0])
    ax.plot([-1, len(labels)], [1.0] * 2, color="black", label="Enzyme/MLIR")
    for i, (method, result) in enumerate(zip(methods, results)):
        # offset_x = x + width * (i - 1)
        offset_x = x + width * (i - 0.5)
        ax.bar(offset_x, result - ones, width, bottom=ones, label=method)
        for point, maybenan in zip(offset_x, result):
            if np.isnan(maybenan):
                ax.plot(point, 1.0, "x", color="red")
            elif maybenan < powers[0]:
                s = f"{maybenan:.1e}"
                m, e = s.split("e")
                ax.text(
                    point - (0.25 if i == 1 else -0.12),
                    # point - 0.25,
                    powers[0] + 0.01,
                    f"${m:s}\\times 10^{{{int(e):d}}}$",
                    math_fontfamily="stix",
                    size="small",
                    horizontalalignment="left",
                    rotation="vertical",
                )

    if action_text:
        for text, (tx, ty) in [
            ("Up to 50$\\times$ smaller memory footprint", (4, 0.6))
        ]:
            ax.text(
                tx,
                ty,
                text,
                math_fontfamily="stix",
                size=30,
                color="crimson",
                horizontalalignment="center",
            )
    ax.legend()

    # ax.set_title(
    #     "Relative Memory Usage Reduction of LAGrad vs Enzyme and PyTorch (Higher is Better)"
    # )
    ax.set_ylabel("Relative Memory Usage Reduction")
    ax.set_yscale("log")
    # ax.set_yticks(powers, powers)
    ax.grid(axis="x")
    ylabels = [(f"1/{int(1/x)}" if x < 1 else f"{int(x)}") for x in powers]
    ax.set_yticks(powers, ylabels)
    ax.set_ylim(powers[0], powers[-1])
    ax.set_xlim((-0.8, x[-1] + 0.8))
    ax.set_xticks(x, labels)
    set_hierarchical_xlabels(x, labels, ax)
    fig.tight_layout()
    set_margins()
    if save:
        plt.savefig("testfigure.pdf")
    else:
        plt.show()


def plot_hand():
    data = lstm_memory()
    print(data)
    print(stats.gmean(data[0] / data[-1]))
    return
    hand_data = get_gmm_data()
    hand_mem = gmm_memory()
    # print(stats.gmean((hand_mem[0] / hand_mem[-1])[:-1]))
    # print(stats.gmean((hand_mem[0])))
    # print(hand_mem)
    # return
    hand_data = np.delete(hand_data, (2,), axis=0)
    hand_mem = np.delete(hand_mem, (2,), axis=0)
    methods = ["LAGrad", "Enzyme", "PyTorch"]
    width = 0.25
    x = np.arange(3)
    ones = np.ones((3,))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=dims)
    labels = ["small", "medium", "large"]
    ax.plot([-1, len(labels)], [1.0] * 2, color="black")
    ax2.plot([-1, len(labels)], [1.0] * 2, color="black")
    ax.set_xlim((-0.8, x[-1] + 0.8))
    ax2.set_xlim((-0.8, x[-1] + 0.8))
    for i, (method, (result, mem)) in enumerate(zip(methods, zip(hand_data, hand_mem))):
        offset_x = x + width * (i - 1)
        # offset_x = x + width * (i - 0.5)
        ax.bar(offset_x, result, width, label=method)
        ax2.bar(offset_x, mem, width, label=method)
        # ax.bar(offset_x, result - ones, width, bottom=ones, label=method)
        for point, maybenan in zip(offset_x, result):
            if np.isnan(maybenan):
                ax.plot(point, 0.05, "x", color="red")
                ax2.plot(point, 0.2, "x", color="red")
    # ax.set_yscale("log")
    fig.suptitle(
        "Gaussian Mixture Model Performance and Memory Usage (Higher is Better)"
    )
    # fig.tight_layout()
    # ax.set_title("Speedup (Higher is Better)")
    # ax2.set_title("Relative Memory Usage Reduction (Higher is Better)")
    ax.set_ylabel("Speedup")
    ax2.set_ylabel("Relative Memory Reduction")
    ax.grid(axis="x")
    ax2.grid(axis="x")
    ax.set_xticks(x, labels)
    ax2.set_xticks(x, labels)
    ax.legend()
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    plot_stacks(save=args.save)
    # plot_mem_vs(save=args.save, action_text=False)
    plot_vs_sota(save=args.save, action_text=False)
    # plot_hand()
