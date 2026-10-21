"""Part 1: Distributed K-Way Log Stream Merger (divide-and-conquer, lazy loading)."""
import csv
from datetime import datetime, timedelta

EPOCH = datetime(1970, 1, 1)


def parse_timestamp(date, time):
    """Convert Date ('2015-10-18') and Time ('18:01:51,775') columns to epoch milliseconds."""
    dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S,%f")
    return (dt - EPOCH) // timedelta(milliseconds=1)


def read_log_stream(path, source):
    """Lazily yield (timestamp_ms, source, row) one line at a time from a log file."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            # row[2] = Date, row[3] = Time
            yield parse_timestamp(row[2], row[3]), source, row


def merge_two(left, right):
    """Lazily merge two sorted streams, pulling one entry at a time from each."""
    a = next(left, None)
    b = next(right, None)
    while a is not None and b is not None:
        # <= keeps ties in left-stream order (stable merge)
        if a[0] <= b[0]:
            yield a
            a = next(left, None)
        else:
            yield b
            b = next(right, None)
    # drain whichever stream still has entries
    while a is not None:
        yield a
        a = next(left, None)
    while b is not None:
        yield b
        b = next(right, None)


def k_way_merge(streams):
    """Recursively merge a list of sorted streams into one sorted stream."""
    if len(streams) == 0:
        return iter(())
    if len(streams) == 1:  # base case: a single stream is already sorted
        return streams[0]
    # dividing the streams into two groups
    mid = len(streams) // 2
    left = k_way_merge(streams[:mid])
    right = k_way_merge(streams[mid:])
    # combine: final two-way merge of the two groups
    return merge_two(left, right)


def merge_log_files(paths, output_path):
    """Merge the log files into output_path, writing each entry as it is produced.
    Returns the number of entries written."""
    with open(paths[0], newline="") as f:
        header = next(csv.reader(f))
    streams = [read_log_stream(p, i) for i, p in enumerate(paths)]
    count = 0
    with open(output_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["Source", "Timestamp"] + header)
        for ts, source, row in k_way_merge(streams):
            writer.writerow([source, ts] + row)
            count += 1
    return count
