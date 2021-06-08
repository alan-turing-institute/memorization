# -*- coding: utf-8 -*-

"""
Generate figure showing the average nearest neighbor distance ratio for bins of 
the memorization score.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import argparse
import numpy as np

from analysis_utils import dict2tex


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nns", help="Nearest neighbor file (.npz)", required=True
    )
    parser.add_argument(
        "--results", help="Memorization results (.npz)", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Output file to write to (.tex)", required=True
    )
    return parser.parse_args()


def make_tex(x, y, yerr, xmin, xmax):
    tex = []

    tex.append("\\documentclass[10pt,preview=true]{standalone}")
    tex.append("\\pdfinfoomitdate=1")
    tex.append("\\pdftrailerid{}")
    tex.append("\\pdfsuppressptexinfo=1")
    tex.append("\\pdfinfo{ /Creator () /Producer () }")
    tex.append("\\usepackage[utf8]{inputenc}")
    tex.append("\\usepackage[T1]{fontenc}")
    tex.append("\\usepackage{pgfplots}")
    tex.append(
        "\\pgfplotsset{compat=newest, scaled y ticks=false, scaled x ticks=false}"
    )

    tex.append("\\definecolor{MyBlue}{HTML}{004488}")
    tex.append("\\definecolor{MyYellow}{HTML}{DDAA33}")
    tex.append("\\definecolor{MyRed}{HTML}{BB5566}")

    tex.append("\\begin{document}")
    tex.append("\\begin{tikzpicture}")

    ymin = 0.75
    ymax = 1.5
    fontsize = "\\normalsize"

    axis_opts = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "scale only axis": None,
        "xlabel": "Memorization score ($M_i$)",
        "ylabel": "Average distance ratio ($\\rho_i$)",
        "width": "8cm",
        "height": "5cm",
        "xlabel style": {"font": fontsize},
        "ylabel style": {"font": fontsize},
        "xticklabel style": {"font": fontsize},
        "yticklabel style": {"font": fontsize},
        "/pgf/number format/.cd, 1000 sep={}": None,
    }
    plot_opts = {
        "only marks": None,
        "mark options": {"scale": 0.75, "draw": "MyBlue", "fill": "MyBlue"},
        "error bars/.cd": None,
        "y dir": "both",
        "y explicit relative": None,
        "error bar style={gray}": None,
    }
    horz_plot_opts = {"forget plot": None, "dashed": None, "draw": "gray"}

    tex.append(f"\\begin{{axis}}[{dict2tex(axis_opts)}]")

    tex.append(f"\\addplot[{dict2tex(plot_opts)}] coordinates {{%")

    for xx, yy, err in zip(x, y, yerr):
        tex.append(f"({xx}, {yy}) +- (0, {err})")
    tex.append("};")

    tex.append(f"\\addplot [{dict2tex(horz_plot_opts)}] coordinates {{%")
    tex.append(f"({xmin}, 1.0) ({xmax}, 1.0)")
    tex.append("};")

    tex.append("\\end{axis}")
    tex.append("\\end{tikzpicture}")
    tex.append("\\end{document}")
    return tex


def main():
    args = parse_args()

    results = np.load(args.results, allow_pickle=True)
    nns = np.load(args.nns)

    metadata = results["metadata"][()]
    dataset = metadata["dataset"]

    ratio = nns["ratio"]
    M = results["M"][:, -1]
    assert ratio.shape == M.shape

    if dataset == "CelebA":
        binwidth = 50
        M_min = -550
        M_max = 3200
    elif dataset == "BinarizedMNIST" and metadata["learning_rate"] == 1e-3:
        binwidth = 2
        M_min = -20
        M_max = 130
    elif dataset == "BinarizedMNIST" and metadata["learning_rate"] == 1e-4:
        binwidth = 2
        M_min = -15
        M_max = 60
    else:
        raise NotImplementedError

    n_bins = int((M_max - M_min) // binwidth + 1)

    bins = [M_min + binwidth * i for i in range(n_bins)]

    mem_mid = []
    avg_ratio = []
    se_ratio = []
    for lb, ub in zip(bins[:-1], bins[1:]):
        r = ratio[np.logical_and(M > lb, M <= ub)]
        if not len(r):
            continue
        mem_mid.append(ub - binwidth / 2)
        avg_ratio.append(np.mean(r))
        se_ratio.append(np.std(r) / np.sqrt(len(r)))

    ci = 1.96 * np.array(se_ratio)

    tex = make_tex(mem_mid, avg_ratio, ci, M_min, M_max)
    with open(args.output, "w") as fp:
        fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
