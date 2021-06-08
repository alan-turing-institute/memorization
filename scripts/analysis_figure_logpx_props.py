#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate figure showing the proportion of regular and highly memorized 
observations in bins of the log p(x) values.

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
        "-i",
        "--input",
        help="NumPy compressed archive with summary results",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file to write to (.tex)",
        required=True
    )
    return parser.parse_args()


def make_tex(
    midpoints,
    props_reg,
    props_high,
    binwidth,
):
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

    fontsize = "\\normalsize"

    axis_opts = {
        "xmin": min(midpoints) - binwidth / 2,
        "xmax": max(midpoints) + binwidth / 2,
        "ymin": 0,
        "ymax": 1,
        "bar width": binwidth,
        "scale only axis": None,
        "xlabel": "$\\log P_{\mathcal{A}}(\\mathbf{x} \,|\, \mathcal{D})$",
        "ylabel": "Proportion",
        "width": "8cm",
        "height": "6cm",
        "ybar stacked": None,
        "xlabel style": {"font": fontsize},
        "ylabel style": {"font": fontsize},
        "xticklabel style": {"font": "\\small"},
        "yticklabel style": {"font": "\\small"},
        "legend pos": "south east",
        "legend style": {"font": fontsize},
        "legend cell align": "left",
        "every node near coord/.append style": {
            "yshift": "0pt",
            "black": None,
            "opacity": 1,
            "font": "\\tiny",
            "check for zero/.code": {
                "\pgfkeys{/pgf/fpu=true}": None,
                "\pgfmathparse{\pgfplotspointmeta-0.05}": None,
                "\pgfmathfloatifflags{\pgfmathresult}{-}{%": None,
                "\pgfkeys{/tikz/coordinate}": None,
                "}{}": None,
                "\pgfkeys{/pgf/fpu=false}": None,
            },
            "check for zero": None,
            "/pgf/number format/fixed zerofill": None,
            "/pgf/number format/fixed": None,
            "/pgf/number format/precision": 2,
        },
        "/pgf/number format/1000 sep": "",
    }

    plot_opts_1 = {"fill": "MyBlue", "fill opacity": 0.5}
    plot_opts_2 = {
        "fill": "MyYellow",
        "fill opacity": 0.5,
        "nodes near coords": None,
    }

    tex.append(f"\\begin{{axis}}[{dict2tex(axis_opts)}]")

    tex.append(f"\\addplot[{dict2tex(plot_opts_1)}] coordinates {{%")
    for x, y in zip(midpoints, props_reg):
        tex.append(f"({x}, {y})")
    tex.append("};")
    tex.append("\\addlegendentry{Regular}")

    tex.append(f"\\addplot[{dict2tex(plot_opts_2)}] coordinates {{%")
    for x, y in zip(midpoints, props_high):
        tex.append(f"({x}, {y})")
    tex.append("};")
    tex.append("\\addlegendentry{High mem.}")

    tex.append("\\end{axis}")
    tex.append("\\end{tikzpicture}")
    tex.append("\\end{document}")
    return tex


def main():
    args = parse_args()

    results = np.load(args.input, allow_pickle=True)
    metadata = results["metadata"][()]
    dataset = metadata["dataset"]

    U = results["U"][:, -1]
    M = results["M"][:, -1]
    q = 0.95

    if dataset == "CelebA":
        binwidth = 500
        U_min = -17_000
        U_max = -11_000
    elif dataset == "BinarizedMNIST" and metadata["learning_rate"] == 1e-3:
        binwidth = 10
        U_min = -150
        U_max = -30
    elif dataset == "BinarizedMNIST" and metadata["learning_rate"] == 1e-4:
        binwidth = 10
        U_min = -160
        U_max = -30
    else:
        raise NotImplementedError

    n_bins = (U_max - U_min) // binwidth + 1
    bins = (
        [-float("inf")]
        + [U_min + binwidth * i for i in range(n_bins)]
        + [float("inf")]
    )

    heights_reg = []
    heights_high = []
    midpoints = []

    for lb, ub in zip(bins[:-1], bins[1:]):
        midpoints.append(ub - binwidth / 2)
        in_bin = np.logical_and(U > lb, U < ub)
        count_reg = np.sum(np.logical_and(in_bin, M <= np.quantile(M, q)))
        count_high = np.sum(np.logical_and(in_bin, M > np.quantile(M, q)))
        heights_reg.append(count_reg)
        heights_high.append(count_high)

    midpoints[-1] = midpoints[-2] + binwidth

    heights_reg = np.array(heights_reg)
    heights_high = np.array(heights_high)
    midpoints = np.array(midpoints)

    props_reg = heights_reg / (heights_reg + heights_high)
    props_high = heights_high / (heights_reg + heights_high)

    tex = make_tex(
        midpoints,
        props_reg,
        props_high,
        binwidth,
    )
    with open(args.output, "w") as fp:
        fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
