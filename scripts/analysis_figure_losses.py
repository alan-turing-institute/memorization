#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate figure showing the evolution of the train/test loss for two models 
trained with different learning rates.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import argparse
import gzip
import json
import matplotlib.pyplot as plt

from collections import defaultdict

from typing import Dict

from analysis_utils import Line
from analysis_utils import dict2tex


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr3-file",
        help="Result file with learning rate 1e-3",
        required=True,
    )
    parser.add_argument(
        "--lr4-file",
        help="Result file with learning rate 1e-4",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file to write to (.tex)",
    )
    return parser.parse_args()


def parse_full(filename: str) -> Dict[int, Dict[int, float]]:

    params = defaultdict(set)
    meta_keys_max = {"dataset": 1, "model": 1, "seed": 1}

    with gzip.open(filename, "rb") as fp:
        contents = fp.read()
        data = json.loads(contents.decode("utf-8"))

    metadata = data["meta"]
    results = data["results"]

    # Ensure we're not mixing datasets/models/seeds
    for key in meta_keys_max:
        params[key].add(metadata[key])
        assert len(params[key]) <= meta_keys_max[key]

    train_loss = results["losses"]["train"]
    test_loss = results["losses"]["test"]

    return train_loss, test_loss


def make_tex(epochs, lr3_train, lr3_test, lr4_train, lr4_test):
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
        "ymax": 126,
        "xlabel": "Epochs",
        "ylabel": "Loss",
        "scale only axis": None,
        "width": "6cm",
        "height": "8cm",
        "xlabel style": {"font": fontsize},
        "ylabel style": {"font": fontsize},
        "xticklabel style": {"font": fontsize},
        "yticklabel style": {"font": fontsize},
        "legend style": {"font": fontsize},
        "legend cell align": "left",
    }

    tex.append(f"\\begin{{axis}}[{dict2tex(axis_opts)}]")

    data = [lr3_train, lr3_test, lr4_train, lr4_test]
    styles = [
        ", ".join(["solid", thickness, "MyBlue"]),
        ", ".join(["densely dashed", thickness, "MyBlue"]),
        ", ".join(["solid", thickness, "MyYellow"]),
        ", ".join(["densely dashed", thickness, "MyYellow"]),
    ]
    labels = [
        "$\\eta = 10^{-3}$, train",
        "$\\eta = 10^{-3}$, test",
        "$\\eta = 10^{-4}$, train",
        "$\\eta = 10^{-4}$, test",
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


def show_plot(epochs, lr3_train, lr3_test, lr4_train, lr4_test):
    plt.plot(epochs, lr3_train, c="tab:blue", label="$\eta = 10^{-3}$, train")
    plt.plot(
        epochs,
        lr3_test,
        c="tab:blue",
        ls="--",
        label="$\eta = 10^{-3}$, test",
    )
    plt.plot(
        epochs, lr4_train, c="tab:orange", label="$\eta = 10^{-4}$, train"
    )
    plt.plot(
        epochs,
        lr4_test,
        c="tab:orange",
        ls="--",
        label="$\eta = 10^{-4}$, test",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def main():
    args = parse_args()

    lr3_train, lr3_test = parse_full(args.lr3_file)
    lr4_train, lr4_test = parse_full(args.lr4_file)

    assert len(set(map(len, [lr3_train, lr3_test, lr4_train, lr4_test]))) == 1

    burn = 5
    L = len(lr3_train)
    epochs = list(range(burn + 1, L + 1))

    lr3_train = lr3_train[burn:]
    lr3_test = lr3_test[burn:]
    lr4_train = lr4_train[burn:]
    lr4_test = lr4_test[burn:]

    if args.output is None:
        show_plot(epochs, lr3_train, lr3_test, lr4_train, lr4_test)
    else:
        tex = make_tex(epochs, lr3_train, lr3_test, lr4_train, lr4_test)
        with open(args.output, "w") as fp:
            fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
