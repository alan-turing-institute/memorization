# -*- coding: utf-8 -*-

"""Shared utilities

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import numpy as np

from typing import Any
from typing import Dict

from scipy.special import logsumexp


def logmeanexp(x):
    x = list(x)
    n = len(x)
    return -np.log(n) + logsumexp(x)


def merge_metadata(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    for key in d2:
        if not key in d1:
            if isinstance(d2[key], dict):
                d1[key] = merge_metadata({}, d2[key])
            else:
                d1[key] = set()
                d1[key].add(d2[key])
        else:
            if isinstance(d2[key], dict) and isinstance(d1[key], dict):
                merge_metadata(d1[key], d2[key])
            elif isinstance(d2[key], dict):
                raise ValueError
            else:
                d1[key].add(d2[key])
    return d1


def clean_metadata(d: Dict[str, Any]) -> Dict[str, Any]:
    new_d = {}
    for key in d:
        if isinstance(d[key], dict):
            new_d[key] = clean_metadata(d[key])
            continue
        l = list(d[key])
        if len(l) == 1:
            new_d[key] = l[0]
        else:
            new_d[key] = sorted(l)

    return new_d
