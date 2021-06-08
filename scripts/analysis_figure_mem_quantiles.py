#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate figure showing the evolution of quantiles of the memorization score 
for models trained with two different learning rates.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from analysis_utils import Line
from analysis_utils import dict2tex


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr3-file",
        help="NumPy compressed archive with results for learning rate 1e-3",
        required=True,
    )
    parser.add_argument(
        "--lr4-file",
        help="NumPy compressed archive with results for learning rate 1e-4",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file to write to (.tex)",
    )
    return parser.parse_args()


def make_tex(epochs, M3, M4):
    tex = []

    tex.append("\\documentclass[10pt,preview=true]{standalone}")
    tex.append("\\pdfinfoomitdate=1")
    tex.append("\\pdftrailerid{}")
    tex.append("\\pdfsuppressptexinfo=1")
    tex.append("\\pdfinfo{ /Creator () /Producer () }")
    tex.append("\\usepackage[utf8]{inputenc}")
    tex.append("\\usepackage[T1]{fontenc}")
    tex.append("\\usepackage{pgfplots}")
    tex.append("\\pgfplotsset{compat=newest}")

    tex.append("\\definecolor{MyBlue}{HTML}{004488}")
    tex.append("\\definecolor{MyYellow}{HTML}{DDAA33}")
    tex.append("\\definecolor{MyRed}{HTML}{BB5566}")

    tex.append("\\begin{document}")
    tex.append("\\begin{tikzpicture}")

    fontsize = "\\normalsize"
    thickness = "ultra thick"

    axis_opts = {
        "xmin": 0,
        "xmax": 107,
        "ymin": 8,
        "ymax": 55,
        "scale only axis": None,
        "xlabel": "Epochs",
        "ylabel": "Quantile value",
        "width": "6cm",
        "height": "8cm",
        "xlabel style": {"font": fontsize},
        "ylabel style": {"font": fontsize},
        "xticklabel style": {"font": fontsize},
        "yticklabel style": {"font": fontsize},
        "legend pos": "north west",
        "legend style": {"font": fontsize},
        "legend cell align": "left",
    }

    tex.append(f"\\begin{{axis}}[{dict2tex(axis_opts)}]")

    data = [
        np.quantile(M3, 0.95, axis=0),
        np.quantile(M3, 0.999, axis=0),
        np.quantile(M4, 0.95, axis=0),
        np.quantile(M4, 0.999, axis=0),
    ]
    styles = [
        ", ".join(["solid", thickness, "MyBlue"]),
        ", ".join(["densely dashed", thickness, "MyBlue"]),
        ", ".join(["solid", thickness, "MyYellow"]),
        ", ".join(["densely dashed", thickness, "MyYellow"]),
    ]
    labels = [
        "$\\eta = 10^{-3}$, $q = 0.95$",
        "$\\eta = 10^{-3}$, $q = 0.999$",
        "$\\eta = 10^{-4}$, $q = 0.95$",
        "$\\eta = 10^{-4}$, $q = 0.999$",
    ]

    lines = [
        Line(xs=epochs, ys=ys, style=style, label=label)
        for ys, style, label in zip(data, styles, labels)
    ]
    for line in lines:
        tex.append(f"\\addplot [{line.style}] table {{%")
        tex.extend([f"{x} {y}" for x, y in zip(line.xs, line.ys)])
        tex.append("};")
        tex.append(f"\\addlegendentry{{{line.label}}}")

    tex.append("\\end{axis}")
    tex.append("\\end{tikzpicture}")
    tex.append("\\end{document}")
    return tex


def show_plot(epochs, M3, M4):
    quantiles = [0.95, 0.999]

    plt.figure()
    plt.plot(
        epochs,
        np.quantile(M3, 0.95, axis=0),
        ls="-",
        color="tab:blue",
        label="$\\eta = 10^{-3}$, q = 0.95",
    )
    plt.plot(
        epochs,
        np.quantile(M3, 0.999, axis=0),
        ls="--",
        color="tab:blue",
        label="$\\eta = 10^{-3}$, q = 0.999",
    )

    plt.plot(
        epochs,
        np.quantile(M4, 0.95, axis=0),
        ls="-",
        color="tab:orange",
        label="$\\eta = 10^{-4}$, q = 0.95",
    )
    plt.plot(
        epochs,
        np.quantile(M4, 0.999, axis=0),
        ls="--",
        color="tab:orange",
        label="$\\eta = 10^{-4}$, q = 0.999",
    )
    plt.ylim(ymax=50)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Memorization score")

    plt.show()


def main():
    args = parse_args()

    bmnist3 = np.load(args.lr3_file)
    bmnist4 = np.load(args.lr4_file)

    M3 = bmnist3["M"]
    M4 = bmnist4["M"]

    epochs = np.array(list(range(5, 105, 5)))

    if args.output is None:
        show_plot(epochs, M3, M4)
    else:
        tex = make_tex(epochs, M3, M4)
        with open(args.output, "w") as fp:
            fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
