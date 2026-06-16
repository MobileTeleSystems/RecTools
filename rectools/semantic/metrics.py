import typing as tp
from collections import Counter


def coverage_k(all_topk_items: tp.List[tp.List[int]], k: int, num_items: int) -> float:
    """Calculate fraction of catalog items that appear in at least one user's top-K.

    Parameters
    ----------
    all_topk_items : tp.List[tp.List[int]]
        tp.List of all users' top-Ks
    k : int
        Users' top-K size
    num_items : int
        Number of unique items in dataset

    Returns
    -------
    float
        Coverage@k.
    """
    recommended = set()
    for items in all_topk_items:
        for item in items[:k]:
            recommended.add(item)
    return len(recommended) / num_items


def gini_k(all_topk_items: tp.List[tp.List[int]], k: int) -> float:
    """Gini coefficient over item recommendation frequencies in top-K.

    0 = every item recommended equally often, 1 = all recommendations
    concentrate on a single item.

    Parameters
    ----------
    all_topk_items : _type_
        tp.List of all users' top-Ks
    k : int
        Users' top-K size

    Returns
    -------
    float
        Gini@k.
    """
    counts: Counter[int] = Counter()
    for items in all_topk_items:
        for item in items[:k]:
            counts[item] += 1

    if len(counts) == 0:
        return 0.0

    freqs = sorted(counts.values())
    n = len(freqs)
    cumulative = 0.0
    for i, freq in enumerate(freqs):
        cumulative += (2 * (i + 1) - n - 1) * freq
    total = sum(freqs)
    return cumulative / (n * total)
