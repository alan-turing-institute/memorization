#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate figure showing the distribution of memorization values for models 
trained with two different learning rates.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import argparse
import numpy as np

from fitter import Fitter

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
        "-o", "--output", help="Output file to write to (.tex)", required=True
    )
    return parser.parse_args()


def make_tex(M3, M4):
    m3 = M3[:, -1]
    m4 = M4[:, -1]
    assert len(m3) == len(m4) == 60_000
    del M3, M4

    tex = []

    pgfplotsset = """
    \\pgfplotsset{compat=newest,%
        /pgf/declare function={
    		gauss(\\x) = 1/sqrt(2*pi) * exp(-0.5 * \\x * \\x);%
    		johnson(\\x,\\a,\\b) = \\b/sqrt(\\x * \\x + 1) * gauss(\\a+\\b*ln(\\x+sqrt(\\x*\\x+1)));%
    		johnsonsu(\\x,\\a,\\b,\\loc,\\scale) = johnson((\\x - \\loc)/\\scale,\\a,\\b)/\\scale;%
    	},
    }
    """

    tex.append("\\documentclass[10pt,preview=true]{standalone}")
    # LuaLaTeX
    tex.append(
        "\\pdfvariable suppressoptionalinfo \\numexpr1+2+8+16+32+64+128+512\\relax"
    )
    tex.append("\\usepackage[utf8]{inputenc}")
    tex.append("\\usepackage[T1]{fontenc}")
    tex.append("\\usepackage{pgfplots}")
    tex.append(pgfplotsset)

    tex.append("\\definecolor{MyBlue}{HTML}{004488}")
    tex.append("\\definecolor{MyYellow}{HTML}{DDAA33}")
    tex.append("\\definecolor{MyRed}{HTML}{BB5566}")

    tex.append("\\begin{document}")
    tex.append("\\begin{tikzpicture}")

    fontsize = "\\normalsize"
    bins = 100
    xmin = -15
    xmax = 35
    ymin = 0
    ymax = 0.12

    axis_opts = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "scale only axis": None,
        "xlabel": "Memorization score",
        "ylabel": "Density",
        "width": "6cm",
        "height": "8cm",
        "ytick": "{0, 0.05, 0.10}",
        "yticklabels": "{0, 0.05, 0.10}",
        "xlabel style": {"font": fontsize},
        "ylabel style": {"font": fontsize},
        "xticklabel style": {"font": fontsize},
        "yticklabel style": {"font": fontsize},
        "legend pos": "north east",
        "legend style": {"font": fontsize},
        "legend cell align": "left",
    }

    tex.append(f"\\begin{{axis}}[{dict2tex(axis_opts)}]")

    thickness = "ultra thick"

    hist_plot_opts = {
        "forget plot": None,
        "draw": "none",
        "fill opacity": 0.3,
        "hist": {
            "bins": bins,
            "density": "true",
            "data min": xmin,
            "data max": xmax,
        },
    }
    line_plot_opts = {
        "domain": f"{xmin}:{xmax}",
        "samples": 201,
        "mark": "none",
        "solid": None,
        thickness: None,
    }
    vert_plot_opts = {
        "forget plot": None,
        "densely dashed": None,
        thickness: None,
    }

    ms = [m3, m4]
    labels = ["$\\eta = 10^{-3}$", "$\\eta = 10^{-4}$"]
    colors = ["MyBlue", "MyYellow"]

    for m, label, color in zip(ms, labels, colors):
        hist_plot_opts["fill"] = color
        line_plot_opts["draw"] = color
        vert_plot_opts["draw"] = color

        tex.append(
            f"\\addplot [{dict2tex(hist_plot_opts)}] table[y index=0] {{%"
        )
        tex.append("data")
        for v in m:
            tex.append(str(v.item()))
        tex.append("};")

        f = Fitter(
            m, distributions=["johnsonsu"], xmin=xmin, xmax=xmax, bins=bins
        )
        f.fit()
        params = f.get_best()
        a, b, loc, scale = params["johnsonsu"]

        tex.append(f"\\addplot [{dict2tex(line_plot_opts)}] {{%")
        tex.append(f"johnsonsu(x, {a}, {b}, {loc}, {scale})")
        tex.append("};")
        tex.append(f"\\addlegendentry{{{label}}}")

        tex.append(f"\\addplot [{dict2tex(vert_plot_opts)}] coordinates {{%")
        tex.append(
            f"({np.quantile(m, 0.95)}, {ymin}) ({np.quantile(m, 0.95)}, {ymax})"
        )
        tex.append("};")

    tex.append("\\end{axis}")
    tex.append("\\end{tikzpicture}")
    tex.append("\\end{document}")
    return tex


def main():
    args = parse_args()

    bmnist3 = np.load(args.lr3_file)
    bmnist4 = np.load(args.lr4_file)

    M3 = bmnist3["M"]
    M4 = bmnist4["M"]

    tex = make_tex(M3, M4)
    with open(args.output, "w") as fp:
        fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
