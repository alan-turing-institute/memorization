# -*- coding: utf-8 -*-

"""
Generate figure showing the distribution of the nearest neighbor distance ratio 
for regular and highly memorized observations.

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
        "--nns", help="Nearest neighbor file (.npz)", required=True
    )
    parser.add_argument(
        "--results", help="Memorization results (.npz)", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Output file to write to (.tex)", required=True
    )
    return parser.parse_args()


def make_tex(rho, M, q=0.95, ymax=4.2):
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
    xmin = 0.6
    xmax = 1.6
    ymin = 0

    ytick = "{" + ", ".join(map(str, range(1 + round(ymax)))) + "}"

    axis_opts = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "scale only axis": None,
        "xlabel": "$\\rho_i$",
        "ylabel": "Density",
        "width": "8cm",
        "height": "5cm",
        "xtick": "{0.75, 1.0, 1.25, 1.5}",
        "xticklabels": "{0.75, 1.0, 1.25, 1.5}",
        "ytick": ytick,
        "yticklabels": ytick,
        "xlabel style": {"font": fontsize},
        "ylabel style": {"font": fontsize},
        "xticklabel style": {"font": fontsize},
        "yticklabel style": {"font": fontsize},
        "legend pos": "north east",
        "legend style": {"font": fontsize},
        "legend cell align": "left",
    }

    tex.append(f"\\begin{{axis}}[{dict2tex(axis_opts)}]")

    thickness = "very thick"

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

    rhos = [rho[M <= q], rho[M > q]]
    labels = ["Regular", "High mem."]
    colors = ["MyBlue", "MyYellow"]

    for r, label, color in zip(rhos, labels, colors):
        hist_plot_opts["fill"] = color
        line_plot_opts["draw"] = color

        tex.append(
            f"\\addplot [{dict2tex(hist_plot_opts)}] table[y index=0] {{%"
        )
        tex.append("data")
        for v in r:
            tex.append(str(v.item()))
        tex.append("};")

        f = Fitter(
            r, distributions=["johnsonsu"], xmin=xmin, xmax=xmax, bins=bins
        )
        f.fit()
        params = f.get_best()
        a, b, loc, scale = params["johnsonsu"]

        tex.append(f"\\addplot [{dict2tex(line_plot_opts)}] {{%")
        tex.append(f"johnsonsu(x, {a}, {b}, {loc}, {scale})")
        tex.append("};")
        tex.append(f"\\addlegendentry{{{label}}}")

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
        ymax = 4.3
    elif dataset == "BinarizedMNIST" and metadata["learning_rate"] == 1e-3:
        ymax = 7.6
    elif dataset == "BinarizedMNIST" and metadata["learning_rate"] == 1e-4:
        ymax = 6.4
    else:
        raise NotImplementedError

    tex = make_tex(ratio, M, q=0.95, ymax=ymax)
    with open(args.output, "w") as fp:
        fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
