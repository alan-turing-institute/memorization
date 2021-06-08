#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate illustrations of observations that have low, medium, or high 
memorization scores.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

from dataset import SplitDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Memorization results (.npz)", required=True
    )
    parser.add_argument(
        "--mode",
        help="Figure to generate",
        choices=["top", "bottom", "middle"],
        required=True,
    )
    parser.add_argument(
        "-o", "--output", help="File to save to (.png)", required=True
    )
    return parser.parse_args()


def plot_dims(n):
    """Given n plots, find a pleasing 2-D arrangement"""
    factors = []
    for i in reversed(range(1, n + 1)):
        if n % i == 0:
            factors.append(i)

    diffs = {}
    for i in range(len(factors)):
        a = factors[i]
        for j in range(i, len(factors)):
            b = factors[j]
            if a * b == n:
                diffs[(a, b)] = abs(a - b)

    return sorted(min(diffs, key=diffs.get))


def make_image(sd, indices, output_filename):
    fig = plt.figure()
    u, v = plot_dims(len(indices))
    grid = ImageGrid(fig, 111, nrows_ncols=(u, v), axes_pad=0.1)
    for ax, idx in zip(grid, indices):
        obs, _, _ = sd.get_item_by_index(idx)
        ax.imshow(obs.permute(1, 2, 0), cmap="Greys_r")
        ax.axis("off")

    fig.savefig(output_filename, bbox_inches="tight")


def main():
    args = parse_args()
    results = np.load(args.input, allow_pickle=True)

    metadata = results["metadata"][()]
    dataset = metadata["dataset"]

    params = {}
    if dataset == "CelebA":
        params = {"resize_64": True}

    sd = SplitDataset(dataset, 0, params=params)
    M = sorted([(m, i) for i, m in enumerate(results["M"][:, -1])])

    n = 15

    if args.mode == "top":
        print("Top:")
        for m, _ in M[-n:]:
            print(m)
        idxs = [i for m, i in M[-n:]]
    elif args.mode == "middle":
        med = np.median([m for m, i in M])
        print(f"Median mem. = {med}")
        dist_median = [(abs(m - med), i) for m, i in M]
        idxs = [i for md, i in sorted(dist_median)[:n]]
        print("Middle:")
        for j in idxs:
            print(next((m for m, i in M if i == j), None))
    elif args.mode == "bottom":
        print("Bottom:")
        for m, _ in M[:n]:
            print(m)
        idxs = [i for m, i in M[:n]]
    make_image(sd, idxs, args.output)


if __name__ == "__main__":
    main()
