#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Summarize result files

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import argparse
import gzip
import json
import numpy as np

from typing import Dict
from typing import List
from typing import Tuple

from collections import defaultdict

from tqdm import tqdm

from dataset import SplitDataset
from utils import clean_metadata
from utils import logmeanexp
from utils import merge_metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-files", help="Result files", nargs="+", required=True
    )
    parser.add_argument(
        "-o", "--output-file", help="Output filename (.npz)", required=True
    )
    return parser.parse_args()


def process_results(
    result_files: List[str],
) -> Tuple[str, Dict[int, Dict[int, float]]]:
    paths = sorted(result_files)

    # maps from iter -> {idx : [logpxs]}
    logpxs_in = defaultdict(lambda: defaultdict(list))
    logpxs_out = defaultdict(lambda: defaultdict(list))
    stores = {"train": logpxs_in, "test": logpxs_out}

    params = defaultdict(set)
    meta_keys_max = {"dataset": 1, "model": 1, "seed": 1}
    all_metadata = {}

    for fname in tqdm(paths):
        with gzip.open(fname, "rb") as fp:
            contents = fp.read()
            data = json.loads(contents.decode("utf-8"))

        metadata = data["meta"]
        results = data["results"]

        merge_metadata(all_metadata, metadata)

        # Ensure we're not mixing datasets/models/seeds
        for key in meta_keys_max:
            params[key].add(metadata[key])
            assert len(params[key]) <= meta_keys_max[key]

        for it in results["logpxs"]:
            for key in results["logpxs"][it]:
                store = stores[key]
                for idx in results["logpxs"][it][key]:
                    value = results["logpxs"][it][key][idx]
                    store[int(it)][int(idx)].append(value)

    assert logpxs_in.keys() == logpxs_out.keys()
    first_it = list(logpxs_in.keys())[0]
    num_its = len(logpxs_in.keys())

    indices = sorted(logpxs_in[first_it].keys())
    assert indices == list(range(len(indices)))

    U = np.zeros((len(indices), num_its))
    V = np.zeros((len(indices), num_its))
    for i, it in tqdm(enumerate(logpxs_in)):
        for idx in indices:
            log_in = logpxs_in[it][idx]
            log_out = logpxs_out[it][idx]
            U[idx, i] = logmeanexp(log_in)
            V[idx, i] = logmeanexp(log_out)

    M = U - V
    metadata = clean_metadata(all_metadata)
    return U, V, M, indices, metadata


def main():
    args = parse_args()
    U, V, M, indices, metadata = process_results(args.result_files)
    sd = SplitDataset(metadata["dataset"], metadata["seed"])

    C = None
    if not sd.name == "CelebA":
        C = np.array([sd.get_item_by_index(idx)[1] for idx in indices])

    np.savez(args.output_file, U=U, V=V, M=M, C=C, metadata=metadata)


if __name__ == "__main__":
    main()
