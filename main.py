"""CMPSC 463 Project 1: runs toy verifications and the dataset experiments for all parts."""
import csv
import os
from datetime import datetime, timedelta

from log_merger import k_way_merge, merge_log_files
from strassen import build_dependency_matrix, naive_multiply, strassen
from burst_detection import describe_bursts, detect_bursts, load_timestamps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MERGED_PATH = os.path.join(BASE_DIR, "merged_log.csv")
LOG_FILES = [os.path.join(DATA_DIR, f"p{i}.csv") for i in range(7)]

# burst detection parameters
B = 20                    # max entries in a base-case partition
T_VALUES = [50, 200, 1000]  # burst window in milliseconds
M_VALUES = [5, 10]          # min events per burst


def fmt_ms(ms):
    """Format epoch milliseconds as a readable time."""
    return (datetime(1970, 1, 1) + timedelta(milliseconds=ms)).strftime("%H:%M:%S.%f")[:-3]


def print_matrix(name, M, labels):
    print(f"\n{name}:")
    print("      " + " ".join(f"{l:>4}" for l in labels))
    for label, row in zip(labels, M):
        print(f"{label:>4}  " + " ".join(f"{v:>4}" for v in row))


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------- toy examples

def toy_merge():
    section("Toy 1: K-Way Merge")
    toy = [[1, 4, 9], [2, 3, 10], [5, 6], [0, 7, 8]]
    for i, s in enumerate(toy):
        print(f"stream {i}: {s}")
    streams = [iter([(t, i, None) for t in s]) for i, s in enumerate(toy)]
    merged = [(t, src) for t, src, _ in k_way_merge(streams)]
    print("merged (timestamp, source):", merged)
    assert [t for t, _ in merged] == sorted(t for s in toy for t in s)
    print("correct: matches sorted()")


def toy_strassen():
    section("Toy 2: Dependency Matrix + Strassen (3x3, not a power of 2)")
    # timeline of sources: 0 -> 1 -> 2 -> 0 -> 2
    timeline = [0, 1, 2, 0, 2]
    print("source timeline:", timeline)
    A = [[0] * 3 for _ in range(3)]
    for x, y in zip(timeline, timeline[1:]):
        if x != y:
            A[x][y] = 1
    labels = ["c0", "c1", "c2"]
    A2 = strassen(A, A)
    A3 = strassen(A2, A)
    print_matrix("A", A, labels)
    print_matrix("A^2", A2, labels)
    print_matrix("A^3", A3, labels)
    assert A2 == naive_multiply(A, A) and A3 == naive_multiply(A2, A)
    print("\ncorrect: matches naive multiplication")


def toy_bursts():
    section("Toy 3: Burst Detection (T=10, M=3, B=4)")
    points = [0, 2, 4, 50, 98, 101, 103, 107, 200, 300]
    print("timestamps:", points)
    bursts = detect_bursts(points, T=10, M=3, B=4)
    for start, end, count in describe_bursts(points, bursts):
        print(f"  burst [{start}, {end}] with {count} events")
    print("total bursts:", len(bursts))


# ---------------------------------------------------------------- dataset experiments

def run_merge():
    section("Part 1: K-Way Merge on Dataset")
    count = merge_log_files(LOG_FILES, MERGED_PATH)
    print(f"merged {count} entries into {os.path.basename(MERGED_PATH)}")
    with open(MERGED_PATH, newline="") as f:
        rows = list(csv.reader(f))[1:]
    # Source, Timestamp, '', LineId, Date, Time, Level, Process, Component, Content, ...
    show = lambda r: f"p{r[0]}  {r[4]} {r[5]}  {r[6]:<5} {r[9][:60]}"
    print("\nFirst 20 entries:")
    for r in rows[:20]:
        print(" ", show(r))
    print("\nLast 20 entries:")
    for r in rows[-20:]:
        print(" ", show(r))


def run_strassen():
    section("Part 2: Dependency Analysis on Dataset")
    k = len(LOG_FILES)
    labels = [f"p{i}" for i in range(k)]
    A = build_dependency_matrix(MERGED_PATH, k)
    A2 = strassen(A, A)
    A3 = strassen(A2, A)
    print_matrix("Dependency matrix A", A, labels)
    print_matrix("Two-hop dependencies A^2", A2, labels)
    print_matrix("Three-hop dependencies A^3", A3, labels)
    assert A2 == naive_multiply(A, A) and A3 == naive_multiply(A2, A)
    print("\nverified against naive multiplication")


def run_bursts():
    section(f"Part 3: Burst Detection on Dataset (B={B})")
    points = load_timestamps(MERGED_PATH)
    print(f"{'M':>3} {'T (ms)':>7} {'bursts':>7} {'events in bursts':>17}")
    results = {}
    for M in M_VALUES:
        for T in T_VALUES:
            bursts = detect_bursts(points, T, M, B)
            results[(M, T)] = bursts
            covered = sum(e - s + 1 for s, e in bursts)
            print(f"{M:>3} {T:>7} {len(bursts):>7} {covered:>17}")

    # list every burst for each case
    for (M, T), bursts in results.items():
        print(f"\nBursts for M={M}, T={T}ms:")
        for start, end, count in describe_bursts(points, bursts):
            print(f"  {fmt_ms(start)} - {fmt_ms(end)}  ({end - start} ms, {count} events)")


if __name__ == "__main__":
    toy_merge()
    toy_strassen()
    toy_bursts()
    run_merge()
    run_strassen()
    run_bursts()
