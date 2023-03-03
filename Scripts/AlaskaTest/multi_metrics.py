"""
Adapting Valentine's metrics, as they work only for a 1-1 comparison, not for matches between more than 2 dataframes.
"""

import math
import collections
from typing import Dict, Tuple, List

def get_tp_fp(matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float],
              golden_standard: Dict,
              n: int = None):
    """
    Calculate the true positive  and false negative numbers of the given matches

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : dict
        A dict that contains the golden standard
    n : int, optional
        The percentage number that we want to consider from the ranked list (matches)
        e.g. (90) for 90% of the matches

    Returns
    -------
    (int, int)
        True positive and false positive counts
    """
    tp = 0
    fp = 0

    if n is not None:
        dict(collections.Counter(matches).most_common(n))

    for match in matches:
        source_table = match[0][0]
        source_column = match[0][1]
        target_table = match[1][0]
        target_column = match[1][1]

        # Did we get that match?
        match_got = False
        for gt in golden_standard:
            if source_table == gt["source_table"] and source_column == gt["source_column"] \
                and target_table == gt["target_table"] and target_column == gt["target_column"]:
                match_got = True
                tp += 1
                break
        if not match_got:
            fp += 1

    #print("1", tp+fp == len(matches))
    return tp, fp

def get_fn(matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float],
           golden_standard: Dict,
           n: int = None):
    """
    Calculate the false positive number of the given matches

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : dict
        A dict that contains the golden standard
    n : int, optional
        The percentage number that we want to consider from the ranked list (matches)
        e.g. (90) for 90% of the matches

    Returns
    -------
    int
        False negative
    """
    fn = 0

    if n is not None:
        dict(collections.Counter(matches).most_common(n))

    for gt in golden_standard:
        match_got = False
        for match in matches:
            source_table = match[0][0]
            source_column = match[0][1]
            target_table = match[1][0]
            target_column = match[1][1]
            if source_table == gt["source_table"] and source_column == gt["source_column"] \
                and target_table == gt["target_table"] and target_column == gt["target_column"]:
                match_got = True
                break
        if not match_got:
            fn += 1
    return fn


def recall(matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float],
           golden_standard: Dict):
    """
    Function that calculates the recall of the matches against the golden standard.

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : dict
        A dict that contains the golden standard

    Returns
    -------
    float
        The recall
    """
    tp, fp = get_tp_fp(matches, golden_standard)
    fn = get_fn(matches, golden_standard)
    #print("2", tp+fn == len(golden_standard), tp+fn, len(golden_standard))
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def precision(matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float],
              golden_standard: Dict):
    """
    Function that calculates the precision of the matches against the golden standard.

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : dict
        A dict that contains the golden standard

    Returns
    -------
    float
        The precision
    """
    tp, fp = get_tp_fp(matches, golden_standard)
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def f1_score(matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float],
             golden_standard: Dict):
    """
    Function that calculates the F1 score of the matches against the golden standard.

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : dict
        A dict that contains the golden standard

    Returns
    -------
    float
        The f1_score
    """
    pr = precision(matches, golden_standard)
    re = recall(matches, golden_standard)
    if pr + re == 0:
        return 0
    return 2 * ((pr * re) / (pr + re))


def recall_at_sizeof_ground_truth(matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float],
                                  golden_standard: Dict):
    """
    Function that calculates the recall at the size of the ground truth.
    e.g. if the size of ground truth size is 10 then only the first 10 matches will be considered for
    the recall calculation

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : dict
        A dict that contains the golden standard

    Returns
    -------
    float
        The recall at the size of ground truth
    """
    tp, fp = get_tp_fp(matches, golden_standard, len(golden_standard))
    fn = get_fn(matches, golden_standard, len(golden_standard))
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)
