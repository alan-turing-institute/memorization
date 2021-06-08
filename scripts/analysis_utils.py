# -*- coding: utf-8 -*-

"""
Utilities for analysis

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

from collections import namedtuple

Line = namedtuple("Line", ["xs", "ys", "style", "label"])


def dict2tex(d):
    items = []
    for key, value in d.items():
        if isinstance(value, dict):
            value = "{\n" + dict2tex(value) + "%\n}"
        if value is None:
            items.append(f"{key}")
        else:
            items.append(f"{key}={value}")
    return ",%\n".join(items)
