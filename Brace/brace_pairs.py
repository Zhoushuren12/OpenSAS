from __future__ import annotations

from collections.abc import Sequence

from .PFSDB import PFSDB, PFSDB_pts
from .SMADB import SMADB, SMADB_pts


BRACE_NODE_PAIRS: tuple[tuple[tuple[int, int, int], ...], ...] = (
    (
        (11, 10010100, 10020108),
        (12, 10010200, 10020108),
        (13, 10010300, 10020308),
        (14, 10010400, 10020308),
    ),
    (
        (21, 11020109, 10030108),
        (22, 11020208, 10030108),
        (23, 11020309, 10030308),
        (24, 11020408, 10030308),
    ),
    (
        (31, 11030109, 10040108),
        (32, 11030208, 10040108),
        (33, 11030309, 10040308),
        (34, 11030408, 10040308),
    ),
    (
        (41, 11040109, 10050108),
        (42, 11040208, 10050108),
        (43, 11040309, 10050308),
        (44, 11040408, 10050308),
    ),
    (
        (51, 11050109, 10060108),
        (52, 11050208, 10060108),
        (53, 11050309, 10060308),
        (54, 11050408, 10060308),
    ),
    (
        (61, 11060109, 10070108),
        (62, 11060208, 10070108),
        (63, 11060309, 10070308),
        (64, 11060408, 10070308),
    ),
    (
        (71, 11070109, 10080108),
        (72, 11070208, 10080108),
        (73, 11070309, 10080308),
        (74, 11070408, 10080308),
    ),
    (
        (81, 11080109, 10090108),
        (82, 11080208, 10090108),
        (83, 11080309, 10090308),
        (84, 11080408, 10090308),
    ),
)


def _expand_minmax(minmax):
    if minmax is None:
        return [None] * len(BRACE_NODE_PAIRS)
    if len(minmax) == 2 and not isinstance(minmax[0], (list, tuple)):
        return [minmax] * len(BRACE_NODE_PAIRS)
    return list(minmax)


def _define_brace_pairs(factory, pts_by_story: Sequence, minmax_by_story=None):
    minmax_items = _expand_minmax(minmax_by_story)
    for story_idx, node_pairs in enumerate(BRACE_NODE_PAIRS):
        pts = pts_by_story[story_idx]
        minmax = minmax_items[story_idx] if story_idx < len(minmax_items) else None
        for spring_id, node_i, node_j in node_pairs:
            factory(spring_id, node_i, node_j, pts, MinMax=minmax)


def define_pfsdb_brace_pairs(T: float, minmax_by_story=None):
    pts_by_story = PFSDB_pts(T)[: len(BRACE_NODE_PAIRS)]
    _define_brace_pairs(PFSDB, pts_by_story, minmax_by_story)
    return pts_by_story


def define_smadb_brace_pairs(T: float):
    bundle = SMADB_pts(T)
    pts_by_story = bundle[: len(BRACE_NODE_PAIRS)]
    minmax1, minmax2 = bundle[-2], bundle[-1]
    minmax_by_story = [minmax1] + [minmax2] * (len(BRACE_NODE_PAIRS) - 1)
    _define_brace_pairs(SMADB, pts_by_story, minmax_by_story)
    return bundle
