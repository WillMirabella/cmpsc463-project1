"""Part 3: Divide-and-Conquer Burst Detection.

A burst is a set of consecutive events with span (max - min timestamp) < T and at least
M events. Bursts never share events, so each event is counted in at most one burst.
Each burst is returned as (start_index, end_index) into the sorted timestamp list.
"""
import csv


def load_timestamps(merged_path):
    """Read the Timestamp column of the merged log (already sorted)."""
    with open(merged_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return [int(row[1]) for row in reader]


def detect_bursts(points, T, M, B):
    """Return the sorted list of non-overlapping bursts in the sorted list points."""
    return _detect(points, 0, len(points), T, M, B)


def _detect(points, lo, hi, T, M, B):
    n = hi - lo
    # base case: partition with at most B entries is a local burst if dense enough
    if n <= B:
        if n >= M and points[hi - 1] - points[lo] < T:
            return [(lo, hi - 1)]
        return []

    # dividing the timeline at the median timestamp
    mid = lo + n // 2
    left = _detect(points, lo, mid, T, M, B)
    right = _detect(points, mid, hi, T, M, B)

    # combine: look for a cross-boundary burst near the median
    cross = _cross_burst(points, lo, mid, hi, left, right, T, M)
    return left + cross + right


def _cross_burst(points, lo, mid, hi, left, right, T, M):
    """Check events within T ms of the boundary that are not already in a burst."""
    boundary = points[mid]
    # skip events already claimed by a burst so bursts stay non-overlapping
    left_limit = left[-1][1] + 1 if left else lo
    right_limit = right[0][0] - 1 if right else hi - 1

    # extend outward from the boundary while events are within T ms of it
    i = mid - 1
    while i >= left_limit and boundary - points[i] < T:
        i -= 1
    j = mid
    while j <= right_limit and points[j] - boundary < T:
        j += 1
    start, end = i + 1, j - 1
    if start > mid - 1 or end < mid:  # need events on both sides
        return []

    # two-pointer scan for the largest window with span < T that crosses the boundary
    best = None
    s = start
    for e in range(mid, end + 1):
        while points[e] - points[s] >= T:
            s += 1
        if s <= mid - 1 and (best is None or e - s > best[1] - best[0]):
            best = (s, e)

    if best and best[1] - best[0] + 1 >= M:
        return [best]
    return []


def describe_bursts(points, bursts):
    """Return (start_ms, end_ms, count) for each burst."""
    return [(points[s], points[e], e - s + 1) for s, e in bursts]
