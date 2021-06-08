# -*- coding: utf-8 -*-

"""Utility to allow reliable restarts of cross-validation runs

The "seed generator" keeps the seeds for the runs so that you can reliably stop 
and restart running the experiments at a specific run/fold. It uses SyncRNG 
because its simple, works the same across platforms, and doesn't interact with 
NumPy/PyTorch/Python. An alternative would be to use the Random class from 
Python's random module.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

from typing import List

from SyncRNG import SyncRNG


class SeedGenerator:
    def __init__(self, root_seed: int):
        self._root_seed = root_seed
        self._rng = SyncRNG(self._root_seed)

    def get_permutation(self, N: int, run: int) -> List[int]:
        run_seed = self._rng.randi() % 10000
        # This uses a separate SyncRNG instance to ensure that there is no
        # dependence on N in the self._rng sequence.
        s = SyncRNG(seed=run_seed)
        indices = list(range(N))
        permutation = s.shuffle(indices)
        return permutation
