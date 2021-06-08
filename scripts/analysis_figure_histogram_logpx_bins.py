#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate histogram of log p(x) for regular and highly memorized observations, 
along with randomly selected example images from different bins.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import argparse
import numpy as np
import os
import torch

from PIL import Image
from functools import reduce

from dataset import SplitDataset

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
    parser.add_argument(
        "--outdir",
        help="Output directory for image files. If omitted, adjacent to output",
    )
    parser.add_argument(
        "--seed", help="Random seed for image selection", type=int
    )
    return parser.parse_args()


def make_tex(
    midpoints,
    heights_reg,
    heights_high,
    binwidth,
    images,
    bottom_lines,
    top_lines,
    ymax=60_000,
    xticks=None,
    yticks=None,
    legend_pos="north east",
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

    xtick = "{" + ", ".join(map(str, xticks)) + "}"

    ytick = list(map(int, map(lambda s: s.replace("k", "000"), yticks)))
    ytick = "{" + ", ".join(map(str, ytick)) + "}"
    yticklabels = "{" + ", ".join(yticks) + "}"

    axis_opts = {
        "xmin": min(midpoints) - binwidth / 2,
        "xmax": max(midpoints) + binwidth / 2,
        "ymin": 0,
        "ymax": ymax,
        "bar width": binwidth,
        "scale only axis": None,
        "xlabel": "$\\log P_{\mathcal{A}}(\\mathbf{x} \,|\, \mathcal{D})$",
        "ylabel": "Count",
        "width": "10cm",
        "height": "6cm",
        "xtick": xtick,
        "ytick": ytick,
        "yticklabels": yticklabels,
        "ybar stacked": None,
        "xlabel style": {"font": fontsize},
        "ylabel style": {"font": fontsize},
        "xticklabel style": {"font": fontsize},
        "yticklabel style": {"font": fontsize},
        "legend pos": legend_pos,
        "legend style": {"font": fontsize},
        "legend cell align": "left",
        "every node near coord/.append style": {
            "yshift": "1pt",
            "black": None,
            "opacity": 1,
            "font": "\\scriptsize",
        },
        "/pgf/number format/.cd, 1000 sep={}": None,
    }

    plot_opts_1 = {"fill": "MyBlue", "fill opacity": 0.5}
    plot_opts_2 = {
        "fill": "MyYellow",
        "fill opacity": 0.5,
        "nodes near coords": None,
        "nodes near coords align": "vertical",
    }
    node_opts = {
        "draw": None,
        # "draw opacity": 0.5,
        "line width": "1.5mm",
        "inner sep": "0pt",
        "anchor": "south",
    }

    tex.append(f"\\begin{{axis}}[{dict2tex(axis_opts)}]")

    tex.append(f"\\addplot[{dict2tex(plot_opts_1)}] coordinates {{%")
    for x, y in zip(midpoints, heights_reg):
        tex.append(f"({x}, {y})")
    tex.append("};")
    tex.append("\\addlegendentry{Regular}")

    tex.append(f"\\addplot[{dict2tex(plot_opts_2)}] coordinates {{%")
    for x, y in zip(midpoints, heights_high):
        tex.append(f"({x}, {y})")
    tex.append("};")
    tex.append("\\addlegendentry{High mem.}")

    for pth, info in images.items():
        node_opts["draw"] = "MyYellow" if info["mem"] else "MyBlue"
        node_opts["yshift"] = f"{(8 + 1.5) * info.get('yshift', 0)}mm"
        node_opts["xshift"] = f"{(8 + 1.5) * info.get('xshift', 0)}mm"
        tex.append(
            f"\\node[{dict2tex(node_opts)}] at ({info['x']}, {info['y']}) {{%"
        )
        tex.append(f"%% U = {info['U']}, M = {info['M']}")
        tex.append(f"\\includegraphics[width=8mm]{{{pth}}}")
        tex.append("};")

    for (a, b), (c, d) in bottom_lines:
        tex.append(
            f"\\draw[densely dashed, gray] ({a}, {b}) to[out=90, in=-90] ({c}, {d});"
        )

    for tl in top_lines:
        texline = "\\draw[densely dashed, gray] "
        x, y = tl.pop(0)
        texline += f"({x}, {y}) to[out=90, in=0]"
        for x, y in tl[:-1]:
            texline += f" ({x}, {y}) to[out=180, in=30]"
        x, y = tl.pop(-1)
        texline += f" ({x}, {y});"
        tex.append(texline)

    tex.append("\\end{axis}")
    tex.append("\\end{tikzpicture}")
    tex.append("\\end{document}")
    return tex


def tensor_to_image(obs: torch.Tensor) -> Image:
    return Image.fromarray(
        np.uint8(255 * obs.permute(1, 2, 0).squeeze().numpy())
    )


def export_image(sd: SplitDataset, idx: int, output_dir, relpath=False) -> str:
    obs = sd.get_item_by_index(idx)[0]
    img = tensor_to_image(obs)
    filename = f"{sd.name}_{idx}.png"
    pth = os.path.join(output_dir, filename)
    img.save(pth)
    return filename if relpath else pth


def random_satisfying(*args, size=1):
    first = args[0]
    assert isinstance(first, np.ndarray)
    init = np.ones_like(first, dtype=bool)
    cond = reduce(np.logical_and, args, init)
    indices = np.where(cond)[0]
    return np.random.choice(indices, size=size, replace=False)


def main():
    args = parse_args()
    np.random.seed(args.seed if args.seed else np.random.randint(10000))

    results = np.load(args.input, allow_pickle=True)
    metadata = results["metadata"][()]
    dataset = metadata["dataset"]
    root_seed = metadata["seed"]
    sd = SplitDataset(dataset, root_seed, params={"resize_64": True})

    outdir = args.outdir if args.outdir else os.path.dirname(args.output)
    relpath = args.outdir is None

    U = results["U"][:, -1]
    M = results["M"][:, -1]
    q = 0.95

    if dataset == "CelebA":
        binwidth = 500
        U_min = -17_000
        U_max = -11_000
        y_max = 60_000
        xticks = list(range(U_min, U_max + 1000, 1000))
        yticks = ["0", "20k", "40k", "60k"]
        legend_pos = "north east"
    elif dataset == "BinarizedMNIST" and metadata["learning_rate"] == 1e-3:
        binwidth = 10
        U_min = -150
        U_max = -30
        y_max = 16000
        xticks = list(range(U_min, U_max + 10, 20))
        yticks = ["0", "4k", "8k", "12k", "16k"]
        legend_pos = "north east"
    elif dataset == "BinarizedMNIST" and metadata["learning_rate"] == 1e-4:
        binwidth = 10
        U_min = -160
        U_max = -30
        y_max = 15000
        xticks = list(range(U_min, U_max + 10, 20))
        yticks = ["0", "5k", "10k", "15k"]
        legend_pos = "north east"
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
        in_bin = np.logical_and(U > lb, U <= ub)
        count_reg = np.sum(np.logical_and(in_bin, M <= np.quantile(M, q)))
        count_high = np.sum(np.logical_and(in_bin, M > np.quantile(M, q)))
        heights_reg.append(count_reg)
        heights_high.append(count_high)

    midpoints[-1] = midpoints[-2] + binwidth

    heights_reg = np.array(heights_reg)
    heights_high = np.array(heights_high)
    midpoints = np.array(midpoints)

    images = {}
    bottom_lines = []
    top_lines = []

    if dataset == "CelebA":
        xpos = U_min
        ypos = 19_000

        # Low logprob, low mem
        idx1, idx2 = random_satisfying(
            U < U_min, M < np.quantile(M, 0.5), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=False, x=xpos, y=ypos, yshift=0, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=False, x=xpos, y=ypos, yshift=1, U=U[idx2], M=M[idx2]
        )

        # Low logprob, high mem
        idx1, idx2 = random_satisfying(
            U < U_min, M > np.quantile(M, 0.999), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=True, x=xpos, y=ypos, yshift=2, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=True, x=xpos, y=ypos, yshift=3, U=U[idx2], M=M[idx2]
        )

        bottom_lines.append([(U_min - binwidth / 2, 4000), (U_min, 18_500)])

        ###
        ###

        xpos = -16_250

        # Med low log prob, low mem (1)
        idx1, idx2 = random_satisfying(
            U < -15_000, U > -15_500, M < np.quantile(M, 0.1), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=False, x=xpos, y=ypos, yshift=0, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=False, x=xpos, y=ypos, yshift=1, U=U[idx2], M=M[idx2]
        )

        # Med low log prob, high mem (1)
        idx1, idx2 = random_satisfying(
            U < -15_000, U > -15_500, M > np.quantile(M, 0.999), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=True, x=xpos, y=ypos, yshift=2, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=True, x=xpos, y=ypos, yshift=3, U=U[idx2], M=M[idx2]
        )

        bottom_lines.append([(-15_000 - binwidth / 2, 5000), (xpos, 18_500)])

        ###
        ###

        xpos = -15_500

        # Central log prob, low mem (1)
        idx1, idx2 = random_satisfying(
            U < -14_000, U > -14_500, M < np.quantile(M, 0.1), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=False, x=xpos, y=ypos, yshift=0, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=False, x=xpos, y=ypos, yshift=1, U=U[idx2], M=M[idx2]
        )

        # Central log prob, high mem (1)
        idx1, idx2 = random_satisfying(
            U < -14_000, U > -14_500, M > np.quantile(M, 0.999), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=True, x=xpos, y=ypos, yshift=2, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=True, x=xpos, y=ypos, yshift=3, U=U[idx2], M=M[idx2]
        )

        bottom_lines.append([(-14_250, 14_000), (xpos, 18_500)])

        ###
        ###

        xpos = -14_750

        # Central log prob, low mem (1)
        idx1, idx2 = random_satisfying(
            U < -13_000, U > -13_500, M < np.quantile(M, 0.1), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=False, x=xpos, y=ypos, yshift=0, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=False, x=xpos, y=ypos, yshift=1, U=U[idx2], M=M[idx2]
        )

        # Central log prob, high mem (1)
        idx1, idx2 = random_satisfying(
            U < -13_000, U > -13_500, M > np.quantile(M, 0.995), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=True, x=xpos, y=ypos, yshift=2, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=True, x=xpos, y=ypos, yshift=3, U=U[idx2], M=M[idx2]
        )

        top_lines.append([(-13250, 53_000), (-14_000, 59_000), (xpos, 58000)])
    elif dataset == "BinarizedMNIST" and metadata["learning_rate"] == 1e-3:
        xpos = -151
        ypos = 5_200

        # Low logprob, low mem
        idx1, idx2 = random_satisfying(
            U < U_min, M < np.quantile(M, 0.5), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=False, x=xpos, y=ypos, yshift=0, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=False, x=xpos, y=ypos, yshift=1, U=U[idx2], M=M[idx2]
        )

        # Low logprob, high mem
        idx1, idx2 = random_satisfying(
            U < U_min, M > np.quantile(M, 0.999), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=True, x=xpos, y=ypos, yshift=2, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=True, x=xpos, y=ypos, yshift=3, U=U[idx2], M=M[idx2]
        )

        bottom_lines.append([(U_min - binwidth / 2, 1200), (U_min, 4_900)])

        ####
        ####

        xpos = -138

        # Med low log prob, low mem (1)
        idx1, idx2 = random_satisfying(
            U < -110, U > -120, M < np.quantile(M, 0.1), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=False, x=xpos, y=ypos, yshift=0, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=False, x=xpos, y=ypos, yshift=1, U=U[idx2], M=M[idx2]
        )

        # Med low log prob, high mem (1)
        idx1, idx2 = random_satisfying(
            U < -110, U > -120, M > np.quantile(M, 0.999), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=True, x=xpos, y=ypos, yshift=2, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=True, x=xpos, y=ypos, yshift=3, U=U[idx2], M=M[idx2]
        )

        bottom_lines.append([(-110 - binwidth / 2, 3000), (xpos, 4_900)])

        ###
        ###

        xpos = -125

        # Central log prob, low mem (1)
        idx1, idx2 = random_satisfying(
            U < -80, U > -90, M < np.quantile(M, 0.1), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=False, x=xpos, y=ypos, yshift=0, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=False, x=xpos, y=ypos, yshift=1, U=U[idx2], M=M[idx2]
        )

        # Central log prob, high mem (1)
        idx1, idx2 = random_satisfying(
            U < -80, U > -90, M > np.quantile(M, 0.995), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=True, x=xpos, y=ypos, yshift=2, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=True, x=xpos, y=ypos, yshift=3, U=U[idx2], M=M[idx2]
        )

        top_lines.append([(-85, 14_000), (-100, 15_700), (xpos, 15_300)])

    elif dataset == "BinarizedMNIST" and metadata["learning_rate"] == 1e-4:
        xpos = -160
        ypos = 4_900

        # Low logprob, low mem
        idx1, idx2 = random_satisfying(
            U < U_min, M < np.quantile(M, 0.5), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=False, x=xpos, y=ypos, yshift=0, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=False, x=xpos, y=ypos, yshift=1, U=U[idx2], M=M[idx2]
        )

        # Low logprob, high mem
        idx1, idx2 = random_satisfying(
            U < U_min, M > np.quantile(M, 0.999), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=True, x=xpos, y=ypos, yshift=2, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=True, x=xpos, y=ypos, yshift=3, U=U[idx2], M=M[idx2]
        )

        bottom_lines.append([(U_min - binwidth / 2, 1200), (U_min, 4_800)])

        ####
        ####

        xpos = -146

        # Med low log prob, low mem (1)
        idx1, idx2 = random_satisfying(
            U < -110, U > -120, M < np.quantile(M, 0.1), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=False, x=xpos, y=ypos, yshift=0, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=False, x=xpos, y=ypos, yshift=1, U=U[idx2], M=M[idx2]
        )

        # Med low log prob, high mem (1)
        idx1, idx2 = random_satisfying(
            U < -110, U > -120, M > np.quantile(M, 0.999), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=True, x=xpos, y=ypos, yshift=2, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=True, x=xpos, y=ypos, yshift=3, U=U[idx2], M=M[idx2]
        )

        bottom_lines.append([(-110 - binwidth / 2, 3800), (xpos, 4_800)])

        ###
        ###

        xpos = -132

        # Central log prob, low mem (1)
        idx1, idx2 = random_satisfying(
            U < -80, U > -90, M < np.quantile(M, 0.1), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=False, x=xpos, y=ypos, yshift=0, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=False, x=xpos, y=ypos, yshift=1, U=U[idx2], M=M[idx2]
        )

        # Central log prob, high mem (1)
        idx1, idx2 = random_satisfying(
            U < -80, U > -90, M > np.quantile(M, 0.995), size=2
        )
        pth1 = export_image(sd, idx1, outdir, relpath=relpath)
        pth2 = export_image(sd, idx2, outdir, relpath=relpath)
        images[pth1] = dict(
            mem=True, x=xpos, y=ypos, yshift=2, U=U[idx1], M=M[idx1]
        )
        images[pth2] = dict(
            mem=True, x=xpos, y=ypos, yshift=3, U=U[idx2], M=M[idx2]
        )

        top_lines.append([(-85, 13_800), (-110, 14_700), (xpos, 14_500)])
    else:
        raise NotImplementedError

    tex = make_tex(
        midpoints,
        heights_reg,
        heights_high,
        binwidth,
        images,
        bottom_lines,
        top_lines,
        ymax=y_max,
        yticks=yticks,
        xticks=xticks,
        legend_pos=legend_pos,
    )
    with open(args.output, "w") as fp:
        fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
