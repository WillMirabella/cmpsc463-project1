<!-- markdownlint-configure-file
{
  "MD013": { "tables": false },
  "MD033": { "allowed_elements": ["sub", "sup"] }
}
-->

# CMPSC 463 Project 1: Divide-and-Conquer Algorithms on Distributed System Logs

**Author:** William Mirabella\
**Course:** CMPSC463  
**Instructor:** Janghoon Yang
**Date:** October 2026

---

## 1. Project Description

This project applies divide-and-conquer algorithms to analyze logs from a
distributed Hadoop system. The dataset is a Hadoop MapReduce job log split into
seven CSV files (`p0.csv` through `p6.csv`). Each file represents one system
component and is sorted by time. Every entry has a `Date` and a `Time`
(millisecond precision, e.g. `18:01:51,775`), a log `Level`, the originating
`Process` and Java `Component`, and the message `Content`.

| File      | Entries  | Main Hadoop class(es)                 | Active period       |
| --------- | -------- | ------------------------------------- | ------------------- |
| p0        | 39       | `ipc.Server`, `CommitterEventHandler` | 18:01:51 – 18:06:28 |
| p1        | 758      | `RMContainerAllocator`, `ipc.Client`  | 18:01:54 – 18:10:54 |
| p2        | 653      | `ipc.Client`, `LeaseRenewer`          | 18:05:27 – 18:10:55 |
| p3        | 134      | `TaskAttemptImpl`, `RackResolver`     | 18:01:53 – 18:06:28 |
| p4        | 53       | `MRAppMaster`, `AsyncDispatcher`      | 18:01:47 – 18:01:53 |
| p5        | 314      | `TaskAttemptListenerImpl`             | 18:02:02 – 18:06:28 |
| p6        | 49       | `ContainerLauncherImpl`               | 18:01:57 – 18:06:28 |
| **Total** | **2000** |                                       | 18:01:47 – 18:10:55 |

### Objectives and Expected Outcomes

1. **K-way merge:** combine the seven sorted streams into one sorted global
   log (`merged_log.csv`). Log entries are read from disk lazily, only when the
   merge needs them.
2. **Dependency analysis:** build a binary component dependency matrix _A_ from
   the merged timeline. Then use Strassen's algorithm, extended to matrices
   whose size is not a power of 2, to compute the two-hop (_A<sup>2</sup>_) and
   three-hop (_A<sup>3</sup>_) dependency matrices.
3. **Burst detection:** recursively split the timeline at the median timestamp
   to find dense regions of activity (bursts). Bursts that cross a partition
   boundary are counted correctly, without overlapping.

### Code Organization

| File                 | Contents                                                |
| -------------------- | ------------------------------------------------------- |
| `log_merger.py`      | Timestamp parsing, lazy reader, two-way and K-way merge |
| `strassen.py`        | Dependency matrix, Strassen's algorithm, naive multiply |
| `burst_detection.py` | Recursive burst detection and cross-boundary combine    |
| `main.py`            | Runs the toy verifications and all dataset experiments  |

Run with `python3 main.py` (Python 3, standard library only).

---

## 2. Algorithms

### 2.1 Distributed K-Way Log Stream Merger Using Divide-and-Conquer

#### 2.1.1 Algorithm Definition

- **Input:** _k_ log streams _S<sub>0</sub>, …, S<sub>k−1</sub>_. Each is a CSV
  file sorted in non-decreasing timestamp order, with _N_ entries in total.
- **Output:** one stream containing all _N_ entries in non-decreasing timestamp
  order, written to `merged_log.csv`. Each output row is tagged with its source
  stream (`Source`) and its timestamp in epoch milliseconds (`Timestamp`).
- **Problem Statement:** merge the _k_ sorted streams into one globally sorted
  stream using divide-and-conquer. No file may be fully loaded into memory: an
  entry is read from disk only when the merge needs it. Entries with equal
  timestamps keep a deterministic order (lower stream index first).

#### 2.1.2 Divide-and-Conquer Design

- **Base Case:** a group of one stream is already sorted, so it is returned
  unchanged. An empty group returns an empty stream.
- **Recursive Strategy:**
  - _Divide:_ split the list of streams into two halves, `streams[:k/2]` and
    `streams[k/2:]`.
  - _Conquer:_ recursively merge each half into one sorted stream.
  - _Combine:_ run a lazy two-way merge on the two resulting streams.

Lazy loading comes from Python generators. Each file is read by a generator
that parses and yields one CSV row at a time. Each two-way merge is also a
generator, holding only the current head entry of each input. Nothing is read
until the output writer asks for the next entry. That request travels down the
merge tree, and exactly one new line is read from exactly one file. The merge
tree has _2k − 1_ nodes, each holding at most one buffered entry, so memory is
**O(k)** no matter how large the files are.

- **Complexity:** the merge tree has depth ⌈log<sub>2</sub> k⌉, and each entry
  passes through one comparison per level, giving **O(N log k)** time. The
  recurrence over the number of streams is
  _T(k) = 2T(k/2) + O(n<sub>k</sub>)_, where _n<sub>k</sub>_ is the number of
  entries in the group.

- **Pseudocode:**

```text
function READ_STREAM(path, source):
    open path; skip header
    for each row in file:                     // one line read per request
        yield (timestamp(row.Date, row.Time), source, row)

function MERGE_TWO(L, R):                     // lazy two-way merge
    a ← next(L);  b ← next(R)
    while a ≠ null and b ≠ null:
        if a.time ≤ b.time:                   // ≤ keeps ties stable
            yield a;  a ← next(L)
        else:
            yield b;  b ← next(R)
    while a ≠ null: yield a; a ← next(L)
    while b ≠ null: yield b; b ← next(R)

function K_WAY_MERGE(streams[0..k-1]):
    if k = 0: return empty stream
    if k = 1: return streams[0]               // base case
    mid ← ⌊k / 2⌋
    left  ← K_WAY_MERGE(streams[0..mid-1])    // divide + conquer
    right ← K_WAY_MERGE(streams[mid..k-1])
    return MERGE_TWO(left, right)             // combine

function MERGE_LOG_FILES(paths, out):
    streams ← [READ_STREAM(p, i) for i, p in paths]
    for entry in K_WAY_MERGE(streams):        // pulls entries one at a time
        write entry to out
```

### 2.2 Strassen's Algorithm for Square Matrix Multiplication of Arbitrary Size

#### 2.2.1 Algorithm Definition

- **Input:** two _n × n_ integer matrices _X_ and _Y_, where _n_ is any
  positive integer (not necessarily a power of 2). In this project _X_ and _Y_
  are powers of the 7 × 7 dependency matrix _A_. _A_ is built from the merged
  timeline: _A\[x\]\[y\] = 1_ if some event from component _y_ immediately
  follows an event from component _x_ (with _x ≠ y_), and _0_ otherwise.
- **Output:** the _n × n_ product _C = XY_. In particular _A<sup>2</sup>_ and
  _A<sup>3</sup>_, whose entry _(x, y)_ counts the dependency paths of length 2
  or 3 from _x_ to _y_.
- **Problem Statement:** compute the matrix product with Strassen's
  divide-and-conquer method, which uses 7 recursive multiplications of
  half-size blocks instead of 8. Matrices of any dimension must be handled.

#### 2.2.2 Divide-and-Conquer Design

- **Base Case:** _n = 1_. Return the scalar product
  _\[\[X<sub>11</sub> · Y<sub>11</sub>\]\]_.
- **Recursive Strategy:**
  - _Odd dimension:_ if _n_ is odd, add one row and one column of zeros to _X_
    and _Y_ to make them _(n+1) × (n+1)_. Multiply recursively, then remove the
    extra row and column from the result. Zero padding does not change the
    original entries of the product: for _i, j &lt; n_,
    _Σ<sub>k</sub> X′<sub>ik</sub>Y′<sub>kj</sub> =
    Σ<sub>k&lt;n</sub> X<sub>ik</sub>Y<sub>kj</sub> + 0_. Padding happens at
    every level, only when needed (e.g. 7 → 8 → 4 → 2 → 1). This adds at most
    one row and column per level, rather than padding all the way to the next
    power of 2 up front.
  - _Divide:_ split each matrix into four _n/2 × n/2_ quadrants
    _A<sub>11</sub>, A<sub>12</sub>, A<sub>21</sub>, A<sub>22</sub>_ and
    _B<sub>11</sub>, B<sub>12</sub>, B<sub>21</sub>, B<sub>22</sub>_.
  - _Conquer:_ compute seven products recursively:
    - M<sub>1</sub> = (A<sub>11</sub> + A<sub>22</sub>)(B<sub>11</sub> +
      B<sub>22</sub>)
    - M<sub>2</sub> = (A<sub>21</sub> + A<sub>22</sub>)B<sub>11</sub>
    - M<sub>3</sub> = A<sub>11</sub>(B<sub>12</sub> − B<sub>22</sub>)
    - M<sub>4</sub> = A<sub>22</sub>(B<sub>21</sub> − B<sub>11</sub>)
    - M<sub>5</sub> = (A<sub>11</sub> + A<sub>12</sub>)B<sub>22</sub>
    - M<sub>6</sub> = (A<sub>21</sub> − A<sub>11</sub>)(B<sub>11</sub> +
      B<sub>12</sub>)
    - M<sub>7</sub> = (A<sub>12</sub> − A<sub>22</sub>)(B<sub>21</sub> +
      B<sub>22</sub>)
  - _Combine:_
    - C<sub>11</sub> = M<sub>1</sub> + M<sub>4</sub> − M<sub>5</sub> +
      M<sub>7</sub>
    - C<sub>12</sub> = M<sub>3</sub> + M<sub>5</sub>
    - C<sub>21</sub> = M<sub>2</sub> + M<sub>4</sub>
    - C<sub>22</sub> = M<sub>1</sub> − M<sub>2</sub> + M<sub>3</sub> +
      M<sub>6</sub>

    Then join the quadrants back into one _n × n_ matrix.

- **Complexity:** _T(n) = 7T(⌈n/2⌉) + O(n<sup>2</sup>)_, so
  _T(n) = O(n<sup>log<sub>2</sub> 7</sup>) ≈ O(n<sup>2.807</sup>)_, compared
  with _O(n<sup>3</sup>)_ for the standard algorithm.

- **Pseudocode:**

```text
function BUILD_DEPENDENCY_MATRIX(merged, k):
    A ← k × k zero matrix
    prev ← null
    for each entry in merged:
        cur ← entry.source
        if prev ≠ null and prev ≠ cur:
            A[prev][cur] ← 1
        prev ← cur
    return A

function STRASSEN(X, Y):
    n ← size(X)
    if n = 1: return [[X[0][0] · Y[0][0]]]        // base case
    if n is odd:                                   // arbitrary-size handling
        X' ← pad X with one zero row and column
        Y' ← pad Y with one zero row and column
        return top-left n × n block of STRASSEN(X', Y')
    h ← n / 2
    split X into A11, A12, A21, A22               // divide
    split Y into B11, B12, B21, B22
    M1 ← STRASSEN(A11 + A22, B11 + B22)
    M2 ← STRASSEN(A21 + A22, B11)
    M3 ← STRASSEN(A11, B12 − B22)
    M4 ← STRASSEN(A22, B21 − B11)
    M5 ← STRASSEN(A11 + A12, B22)
    M6 ← STRASSEN(A21 − A11, B11 + B12)
    M7 ← STRASSEN(A12 − A22, B21 + B22)
    C11 ← M1 + M4 − M5 + M7                       // combine
    C12 ← M3 + M5
    C21 ← M2 + M4
    C22 ← M1 − M2 + M3 + M6
    return [[C11, C12], [C21, C22]]

A2 ← STRASSEN(A, A)
A3 ← STRASSEN(A2, A)
```

### 2.3 Divide-and-Conquer Burst Detection

#### 2.3.1 Algorithm Definition

- **Input:** a sorted list of _N_ event timestamps _P\[0..N−1\]_ (in
  milliseconds) taken from the merged log, and three parameters:
  - _T_: the maximum burst span in milliseconds (strict).
  - _M_: the minimum number of events in a burst.
  - _B_: the maximum number of entries in a base-case partition.
- **Output:** the total number of bursts and the list of bursts. Each burst is
  reported as (start time, end time, event count) and is internally a
  contiguous index range _\[s, e\]_ of _P_.
- **Problem Statement:** a _burst_ is a set of consecutive events whose span
  _P\[e\] − P\[s\]_ is less than _T_ and that contains at least _M_ events.
  Find bursts by recursively dividing the timeline at the median. A base-case
  partition (at most _B_ entries) is a _local burst_ if its span is under _T_
  and it has at least _M_ entries. In the combine step, events within _T_ ms of
  the partition boundary are examined, and if events from both halves together
  satisfy the burst condition, one _cross-boundary burst_ is counted.
  **Bursts must not overlap:** every event belongs to at most one counted
  burst.

#### 2.3.2 Divide-and-Conquer Design

**Split bursts and non-overlap.** A naive combine step would double-count.
Suppose a base case on the right already counted a local burst right next to
the boundary. The events within _T_ of the boundary could then be counted again
as a cross-boundary burst. To stay consistent with counting non-overlapping
bursts, every recursive call returns its bursts as a sorted list of disjoint
index ranges, not just a count. The combine step may only use **unclaimed**
events:

- On the left, only events after the end of the left half's last burst.
- On the right, only events before the start of the right half's first burst.

Because the timeline is sorted and bursts are contiguous ranges, these limits
can be read directly off the two lists without a global "used" array. A
cross-boundary burst is made only of unclaimed events, so it never overlaps a
burst already found in either half. Higher levels then treat it as claimed in
the same way.

**Choosing the cross-boundary window.** The events within _T_ of the boundary
on both sides can together span up to _2T_. A burst needs span &lt; _T_, so the
combine step does not test the whole candidate set at once. It runs a
two-pointer sliding window over the candidates to find the **largest window
with span &lt; T that contains at least one event from each half**. If that
window has at least _M_ events, it is counted as one cross-boundary burst, so
each combine step adds at most one burst, matching the problem statement.

- **Base Case:** a partition _P\[lo..hi−1\]_ with _hi − lo ≤ B_. If it has at
  least _M_ entries and _P\[hi−1\] − P\[lo\] &lt; T_, return _\[(lo, hi−1)\]_
  (one local burst). Otherwise return an empty list.
- **Recursive Strategy:**
  - _Divide:_ split at the median index _mid = lo + ⌊(hi − lo)/2⌋_. The
    boundary timestamp is the median _P\[mid\]_.
  - _Conquer:_ recursively detect bursts in _P\[lo..mid−1\]_ and
    _P\[mid..hi−1\]_.
  - _Combine:_ collect the unclaimed events within _T_ ms of _P\[mid\]_ on each
    side, find the best crossing window, and add one cross-boundary burst if it
    qualifies. Return _left ++ cross ++ right_ (still sorted and disjoint). The
    number of bursts in the interval is the length of this list.
- **Complexity:** the combine step scans at most the events of the current
  interval, so _T(n) = 2T(n/2) + O(n)_, giving **O(N log N)** in the worst
  case. In practice the window near the boundary is small, so the combine step
  is close to constant time.

- **Pseudocode:**

```text
function DETECT(P, lo, hi, T, M, B):       // returns sorted list of bursts
    n ← hi − lo
    if n ≤ B:                               // base case
        if n ≥ M and P[hi−1] − P[lo] < T:
            return [(lo, hi−1)]             // local burst
        return []
    mid ← lo + ⌊n / 2⌋                      // divide at median timestamp
    left  ← DETECT(P, lo, mid, T, M, B)     // conquer
    right ← DETECT(P, mid, hi, T, M, B)
    cross ← CROSS_BURST(P, lo, mid, hi, left, right, T, M)    // combine
    return left ++ cross ++ right

function CROSS_BURST(P, lo, mid, hi, left, right, T, M):
    b ← P[mid]                              // partition boundary
    // skip events already claimed by a burst
    leftLimit  ← lo      if left = []  else last(left).end + 1
    rightLimit ← hi − 1  if right = [] else first(right).start − 1
    i ← mid − 1
    while i ≥ leftLimit and b − P[i] < T: i ← i − 1
    j ← mid
    while j ≤ rightLimit and P[j] − b < T: j ← j + 1
    start ← i + 1;  end ← j − 1
    if start > mid − 1 or end < mid: return []    // need both halves
    best ← null;  s ← start
    for e ← mid to end:                     // two-pointer window
        while P[e] − P[s] ≥ T: s ← s + 1
        if s ≤ mid − 1 and (best = null or e − s > best.e − best.s):
            best ← (s, e)
    if best ≠ null and best.e − best.s + 1 ≥ M:
        return [best]                       // one cross-boundary burst
    return []

bursts ← DETECT(P, 0, N, T, M, B)
total  ← |bursts|
```

---

## 3. Verification Using Toy Examples

All three toy examples are run by `main.py` (functions `toy_merge`,
`toy_strassen`, `toy_bursts`).

### 3.1 Distributed K-Way Log Stream Merger

Input: four sorted streams (timestamps only).

```text
stream 0: [1, 4, 9]
stream 1: [2, 3, 10]
stream 2: [5, 6]
stream 3: [0, 7, 8]
```

Recursion trace:

```text
K_WAY_MERGE([s0, s1, s2, s3])
├── K_WAY_MERGE([s0, s1])
│   ├── s0 (base case)            [1, 4, 9]
│   ├── s1 (base case)            [2, 3, 10]
│   └── MERGE_TWO →               [1, 2, 3, 4, 9, 10]
├── K_WAY_MERGE([s2, s3])
│   ├── s2 (base case)            [5, 6]
│   ├── s3 (base case)            [0, 7, 8]
│   └── MERGE_TWO →               [0, 5, 6, 7, 8]
└── MERGE_TWO →                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

Program output, as (timestamp, source):

```text
[(0, 3), (1, 0), (2, 1), (3, 1), (4, 0), (5, 2), (6, 2), (7, 3), (8, 3),
 (9, 0), (10, 1)]
correct: matches sorted()
```

The result matches Python's built-in `sorted()` on all eleven values, and every
entry keeps the correct source tag.

### 3.2 Fault Dependency Analysis Using Strassen's Matrix Multiplication

Toy timeline of sources: `0 → 1 → 2 → 0 → 2`. The adjacent pairs are (0,1),
(1,2), (2,0), (0,2), which give the dependency matrix _A_:

| A      | c0  | c1  | c2  |
| ------ | --- | --- | --- |
| **c0** | 0   | 1   | 1   |
| **c1** | 0   | 0   | 1   |
| **c2** | 1   | 0   | 0   |

The matrix is 3 × 3, which is not a power of 2, so Strassen pads it to 4 × 4.
It then splits into 2 × 2 blocks, which are split into 1 × 1 base cases, and
finally strips the padding.

| A<sup>2</sup> | c0  | c1  | c2  |
| ------------- | --- | --- | --- |
| **c0**        | 1   | 0   | 1   |
| **c1**        | 1   | 0   | 0   |
| **c2**        | 0   | 1   | 1   |

| A<sup>3</sup> | c0  | c1  | c2  |
| ------------- | --- | --- | --- |
| **c0**        | 1   | 1   | 1   |
| **c1**        | 0   | 1   | 1   |
| **c2**        | 1   | 0   | 1   |

Checking by hand:

- _A<sup>2</sup>\[c0\]\[c0\] = 1_: the single two-hop path c0 → c2 → c0.
- _A<sup>2</sup>\[c1\]\[c0\] = 1_: c1 → c2 → c0.
- _A<sup>2</sup>\[c0\]\[c1\] = 0_: no component reachable from c0 leads to c1
  except c0 itself, and self-loops are excluded.
- _A<sup>3</sup>\[c1\]\[c0\] = 0_: the only paths out of c1 are
  c1 → c2 → c0 → {c1, c2}, so a length-3 path from c1 can end only at c1 or
  c2.

The program also checks both products against the naive _O(n<sup>3</sup>)_
multiplication, and the results match.

### 3.3 Divide-and-Conquer Burst Detection

Parameters: _T = 10 ms, M = 3, B = 4_. Timestamps (indices 0–9):

```text
index:  0  1  2   3   4    5    6    7    8    9
P:      0  2  4  50  98  101  103  107  200  300
```

Recursion trace:

```text
DETECT[0..9]  (n=10, boundary P[5]=101)
├── DETECT[0..4]  (n=5, boundary P[2]=4)
│   ├── DETECT[0..1] = {0,2}          base: n=2 < M       → []
│   ├── DETECT[2..4] = {4,50,98}      base: span 94 ≥ T   → []
│   └── CROSS at 4: candidates {0,2 | 4}, span 4 < 10, 3 events ≥ M
│       → cross-boundary burst (0..2) = {0, 2, 4}
├── DETECT[5..9]  (n=5, boundary P[7]=107)
│   ├── DETECT[5..6] = {101,103}      base: n=2 < M       → []
│   ├── DETECT[7..9] = {107,200,300}  base: span 193 ≥ T  → []
│   └── CROSS at 107: candidates {101,103 | 107}, span 6 < 10, 3 events
│       → cross-boundary burst (5..7) = {101, 103, 107}
└── CROSS at 101: left limit = 3 (after burst 0..2),
    right limit = 4 (before burst 5..7).
    Left candidate {98}, right candidates: none (all claimed)
    → no cross-boundary burst
```

Program output:

```text
  burst [0, 4] with 3 events
  burst [101, 107] with 3 events
total bursts: 2
```

This example shows two properties:

1. **Split bursts are found.** Neither burst lies entirely inside one base
   case, so both were found only by the combine step.
2. **No overlap.** At the top level, a naive combine step would see
   {98, 101, 103, 107} (span 9 &lt; 10, 4 events) and count a third burst,
   reusing three events already counted in {101, 103, 107}. Because the
   claimed range is excluded, this double count is avoided, and the answer of
   2 non-overlapping bursts is correct.

---

## 4. Numerical Experiments Using the Target Dataset

### 4.1 Distributed K-Way Log Stream Merger

All seven files (2000 entries) were merged into `merged_log.csv`. Two columns
were added at the front: `Source` (the file index) and `Timestamp` (epoch ms).

#### First 20 Merged Log Entries

| #   | Source | Time         | Level | Content                                                                 |
| --- | ------ | ------------ | ----- | ----------------------------------------------------------------------- |
| 1   | p4     | 18:01:47,978 | INFO  | Created MRAppMaster for application appattempt_1445144423722_0020_0000… |
| 2   | p4     | 18:01:48,963 | INFO  | Executing with tokens:                                                  |
| 3   | p4     | 18:01:48,963 | INFO  | Kind: YARN*AM_RM_TOKEN, Service: , Ident: (appAttemptId { application*… |
| 4   | p4     | 18:01:49,228 | INFO  | Using mapred newApiCommitter.                                           |
| 5   | p4     | 18:01:50,353 | INFO  | OutputCommitter set in config null                                      |
| 6   | p4     | 18:01:50,509 | INFO  | OutputCommitter is org.apache.hadoop.mapreduce.lib.output.FileOutputCo… |
| 7   | p4     | 18:01:50,556 | INFO  | Registering class org.apache.hadoop.mapreduce.jobhistory.EventType for… |
| 8   | p4     | 18:01:50,556 | INFO  | Registering class org.apache.hadoop.mapreduce.v2.app.job.event.JobEven… |
| 9   | p4     | 18:01:50,556 | INFO  | Registering class org.apache.hadoop.mapreduce.v2.app.job.event.TaskEve… |
| 10  | p4     | 18:01:50,556 | INFO  | Registering class org.apache.hadoop.mapreduce.v2.app.job.event.TaskAtt… |
| 11  | p4     | 18:01:50,572 | INFO  | Registering class org.apache.hadoop.mapreduce.v2.app.commit.CommitterE… |
| 12  | p4     | 18:01:50,572 | INFO  | Registering class org.apache.hadoop.mapreduce.v2.app.speculate.Specula… |
| 13  | p4     | 18:01:50,572 | INFO  | Registering class org.apache.hadoop.mapreduce.v2.app.rm.ContainerAlloc… |
| 14  | p4     | 18:01:50,588 | INFO  | Registering class org.apache.hadoop.mapreduce.v2.app.launcher.Containe… |
| 15  | p4     | 18:01:50,634 | INFO  | Default file system \[hdfs://msra-sa-41:9000\]                          |
| 16  | p4     | 18:01:50,666 | INFO  | Default file system \[hdfs://msra-sa-41:9000\]                          |
| 17  | p4     | 18:01:50,713 | INFO  | Default file system \[hdfs://msra-sa-41:9000\]                          |
| 18  | p4     | 18:01:50,728 | INFO  | Emitting job history data to the timeline server is not enabled         |
| 19  | p4     | 18:01:50,806 | INFO  | Registering class org.apache.hadoop.mapreduce.v2.app.job.event.JobFini… |
| 20  | p4     | 18:01:51,197 | INFO  | loaded properties from hadoop-metrics2.properties                       |

#### Last 20 Merged Log Entries

| #    | Source | Time         | Level | Content                                                                   |
| ---- | ------ | ------------ | ----- | ------------------------------------------------------------------------- |
| 1981 | p1     | 18:10:49,545 | WARN  | Address change detected. Old: msra-sa-41/10.190.173.170:8030 New: msra…   |
| 1982 | p2     | 18:10:50,155 | WARN  | Address change detected. Old: msra-sa-41/10.190.173.170:9000 New: msra…   |
| 1983 | p2     | 18:10:50,155 | WARN  | Failed to renew lease for \[DFSClient_NONMAPREDUCE_1537864556_1\] for 35… |
| 1984 | p1     | 18:10:50,545 | INFO  | Retrying connect to server: msra-sa-41:8030. Already tried 0 time(s); …   |
| 1985 | p1     | 18:10:50,545 | ERROR | ERROR IN CONTACTING RM.                                                   |
| 1986 | p2     | 18:10:51,155 | WARN  | Address change detected. Old: msra-sa-41/10.190.173.170:9000 New: msra…   |
| 1987 | p2     | 18:10:51,155 | WARN  | Failed to renew lease for \[DFSClient_NONMAPREDUCE_1537864556_1\] for 35… |
| 1988 | p1     | 18:10:51,546 | WARN  | Address change detected. Old: msra-sa-41/10.190.173.170:8030 New: msra…   |
| 1989 | p2     | 18:10:52,155 | WARN  | Address change detected. Old: msra-sa-41/10.190.173.170:9000 New: msra…   |
| 1990 | p2     | 18:10:52,155 | WARN  | Failed to renew lease for \[DFSClient_NONMAPREDUCE_1537864556_1\] for 35… |
| 1991 | p1     | 18:10:52,546 | INFO  | Retrying connect to server: msra-sa-41:8030. Already tried 0 time(s); …   |
| 1992 | p1     | 18:10:52,546 | ERROR | ERROR IN CONTACTING RM.                                                   |
| 1993 | p2     | 18:10:53,202 | WARN  | Address change detected. Old: msra-sa-41/10.190.173.170:9000 New: msra…   |
| 1994 | p2     | 18:10:53,202 | WARN  | Failed to renew lease for \[DFSClient_NONMAPREDUCE_1537864556_1\] for 35… |
| 1995 | p1     | 18:10:53,546 | WARN  | Address change detected. Old: msra-sa-41/10.190.173.170:8030 New: msra…   |
| 1996 | p2     | 18:10:54,202 | WARN  | Address change detected. Old: msra-sa-41/10.190.173.170:9000 New: msra…   |
| 1997 | p2     | 18:10:54,202 | WARN  | Failed to renew lease for \[DFSClient_NONMAPREDUCE_1537864556_1\] for 35… |
| 1998 | p1     | 18:10:54,546 | INFO  | Retrying connect to server: msra-sa-41:8030. Already tried 0 time(s); …   |
| 1999 | p1     | 18:10:54,546 | ERROR | ERROR IN CONTACTING RM.                                                   |
| 2000 | p2     | 18:10:55,202 | WARN  | Address change detected. Old: msra-sa-41/10.190.173.170:9000 New: msra…   |

### 4.2 Fault Dependency Analysis Using Strassen's Matrix Multiplication

Components are the seven source files p0–p6 (_n = 7_, not a power of 2).
Self-transitions (_x → x_) are excluded. In each matrix the row is the earlier
component and the column is the component whose event immediately follows.

#### Dependency Matrix A

| A      | p0  | p1  | p2  | p3  | p4  | p5  | p6  |
| ------ | --- | --- | --- | --- | --- | --- | --- |
| **p0** | 0   | 1   | 1   | 1   | 1   | 1   | 0   |
| **p1** | 1   | 0   | 1   | 1   | 0   | 1   | 0   |
| **p2** | 1   | 1   | 0   | 0   | 0   | 1   | 0   |
| **p3** | 1   | 1   | 1   | 0   | 1   | 1   | 1   |
| **p4** | 1   | 0   | 0   | 1   | 0   | 0   | 0   |
| **p5** | 1   | 1   | 1   | 1   | 0   | 0   | 1   |
| **p6** | 1   | 1   | 1   | 1   | 0   | 1   | 0   |

#### Two-Hop Dependency Matrix A<sup>2</sup>

| A<sup>2</sup> | p0  | p1  | p2  | p3  | p4  | p5  | p6  |
| ------------- | --- | --- | --- | --- | --- | --- | --- |
| **p0**        | 5   | 3   | 3   | 3   | 1   | 3   | 2   |
| **p1**        | 3   | 4   | 3   | 2   | 2   | 3   | 2   |
| **p2**        | 2   | 2   | 3   | 3   | 1   | 2   | 1   |
| **p3**        | 5   | 4   | 4   | 5   | 1   | 4   | 1   |
| **p4**        | 1   | 2   | 2   | 1   | 2   | 2   | 1   |
| **p5**        | 4   | 4   | 4   | 3   | 2   | 5   | 1   |
| **p6**        | 4   | 4   | 4   | 3   | 2   | 4   | 2   |

#### Three-Hop Dependency Matrix A<sup>3</sup>

| A<sup>3</sup> | p0  | p1  | p2  | p3  | p4  | p5  | p6  |
| ------------- | --- | --- | --- | --- | --- | --- | --- |
| **p0**        | 15  | 16  | 16  | 14  | 8   | 16  | 6   |
| **p1**        | 16  | 13  | 14  | 14  | 5   | 14  | 5   |
| **p2**        | 12  | 11  | 10  | 8   | 5   | 11  | 5   |
| **p3**        | 19  | 19  | 19  | 15  | 10  | 19  | 9   |
| **p4**        | 10  | 7   | 7   | 8   | 2   | 7   | 3   |
| **p5**        | 19  | 17  | 17  | 16  | 7   | 16  | 8   |
| **p6**        | 19  | 17  | 17  | 16  | 7   | 17  | 7   |

Both _A<sup>2</sup>_ and _A<sup>3</sup>_ computed by Strassen were checked
against the naive triple-loop product, and they match exactly.

### 4.3 Divide-and-Conquer Burst Detection

Base-case size _B = 20_. Three window lengths _T ∈ {50, 200, 1000} ms_ were
tested for two minimum sizes _M ∈ {5, 10}_.

#### Summary of the Six Cases

| M   | T (ms) | Bursts detected | Events in bursts | % of all 2000 events |
| --- | ------ | --------------- | ---------------- | -------------------- |
| 5   | 50     | 18              | 178              | 8.9%                 |
| 5   | 200    | 22              | 263              | 13.2%                |
| 5   | 1000   | 40              | 449              | 22.5%                |
| 10  | 50     | 7               | 99               | 5.0%                 |
| 10  | 200    | 15              | 215              | 10.8%                |
| 10  | 1000   | 22              | 345              | 17.3%                |

Each list below gives start – end, span, and number of events.

#### Case 1: M = 5, T = 50 ms (18 Bursts)

```text
18:01:51.775 - 18:01:51.806  (31 ms, 5 events)
18:01:53.822 - 18:01:53.869  (47 ms, 15 events)
18:01:53.869 - 18:01:53.885  (16 ms, 16 events)
18:01:57.072 - 18:01:57.119  (47 ms, 6 events)
18:01:58.916 - 18:01:58.963  (47 ms, 16 events)
18:02:01.041 - 18:02:01.041  (0 ms, 5 events)
18:02:47.497 - 18:02:47.513  (16 ms, 7 events)
18:03:55.408 - 18:03:55.423  (15 ms, 10 events)
18:03:55.923 - 18:03:55.939  (16 ms, 11 events)
18:03:56.455 - 18:03:56.501  (46 ms, 9 events)
18:03:57.533 - 18:03:57.564  (31 ms, 8 events)
18:04:04.877 - 18:04:04.877  (0 ms, 8 events)
18:04:07.908 - 18:04:07.955  (47 ms, 8 events)
18:04:08.987 - 18:04:08.987  (0 ms, 8 events)
18:04:09.955 - 18:04:10.002  (47 ms, 7 events)
18:04:51.114 - 18:04:51.145  (31 ms, 8 events)
18:06:26.108 - 18:06:26.139  (31 ms, 14 events)
18:06:28.217 - 18:06:28.248  (31 ms, 17 events)
```

#### Case 2: M = 5, T = 200 ms (22 Bursts)

```text
18:01:50.556 - 18:01:50.728  (172 ms, 12 events)
18:01:51.775 - 18:01:51.963  (188 ms, 14 events)
18:01:53.822 - 18:01:53.869  (47 ms, 15 events)
18:01:53.869 - 18:01:53.885  (16 ms, 16 events)
18:01:57.104 - 18:01:57.291  (187 ms, 9 events)
18:01:58.916 - 18:01:58.963  (47 ms, 16 events)
18:02:00.963 - 18:02:01.041  (78 ms, 16 events)
18:02:12.745 - 18:02:12.855  (110 ms, 5 events)
18:02:47.497 - 18:02:47.685  (188 ms, 10 events)
18:03:55.408 - 18:03:55.595  (187 ms, 16 events)
18:03:55.923 - 18:03:56.064  (141 ms, 12 events)
18:03:56.361 - 18:03:56.548  (187 ms, 13 events)
18:03:57.502 - 18:03:57.564  (62 ms, 9 events)
18:04:04.877 - 18:04:04.986  (109 ms, 12 events)
18:04:07.908 - 18:04:08.065  (157 ms, 12 events)
18:04:08.987 - 18:04:09.096  (109 ms, 11 events)
18:04:09.955 - 18:04:10.002  (47 ms, 7 events)
18:04:20.206 - 18:04:20.394  (188 ms, 5 events)
18:04:51.114 - 18:04:51.145  (31 ms, 8 events)
18:06:25.061 - 18:06:25.232  (171 ms, 5 events)
18:06:25.998 - 18:06:26.139  (141 ms, 24 events)
18:06:28.092 - 18:06:28.233  (141 ms, 16 events)
```

#### Case 3: M = 5, T = 1000 ms (40 Bursts)

```text
18:01:50.353 - 18:01:51.322  (969 ms, 19 events)
18:01:51.791 - 18:01:52.728  (937 ms, 15 events)
18:01:53.400 - 18:01:53.775  (375 ms, 16 events)
18:01:53.822 - 18:01:53.869  (47 ms, 15 events)
18:01:53.869 - 18:01:53.885  (16 ms, 16 events)
18:01:53.885 - 18:01:54.791  (906 ms, 16 events)
18:01:56.916 - 18:01:57.447  (531 ms, 21 events)
18:01:58.916 - 18:01:58.963  (47 ms, 16 events)
18:02:00.963 - 18:02:01.041  (78 ms, 16 events)
18:02:05.229 - 18:02:05.870  (641 ms, 5 events)
18:02:12.745 - 18:02:12.855  (110 ms, 5 events)
18:02:47.497 - 18:02:47.841  (344 ms, 15 events)
18:02:49.372 - 18:02:50.216  (844 ms, 5 events)
18:02:52.716 - 18:02:53.685  (969 ms, 5 events)
18:03:03.154 - 18:03:03.795  (641 ms, 6 events)
18:03:09.702 - 18:03:10.561  (859 ms, 5 events)
18:03:13.577 - 18:03:14.155  (578 ms, 5 events)
18:03:24.390 - 18:03:25.281  (891 ms, 5 events)
18:03:45.345 - 18:03:46.282  (937 ms, 5 events)
18:03:55.408 - 18:03:55.939  (531 ms, 16 events)
18:03:55.939 - 18:03:56.455  (516 ms, 15 events)
18:03:56.798 - 18:03:57.564  (766 ms, 14 events)
18:03:59.502 - 18:04:00.049  (547 ms, 5 events)
18:04:04.877 - 18:04:05.830  (953 ms, 16 events)
18:04:07.908 - 18:04:08.877  (969 ms, 19 events)
18:04:08.987 - 18:04:09.955  (968 ms, 16 events)
18:04:10.940 - 18:04:11.924  (984 ms, 12 events)
18:04:13.346 - 18:04:14.315  (969 ms, 8 events)
18:04:19.753 - 18:04:20.550  (797 ms, 7 events)
18:04:34.832 - 18:04:35.816  (984 ms, 6 events)
18:04:44.864 - 18:04:45.754  (890 ms, 8 events)
18:04:50.208 - 18:04:51.114  (906 ms, 16 events)
18:04:51.755 - 18:04:52.552  (797 ms, 12 events)
18:05:56.696 - 18:05:57.024  (328 ms, 7 events)
18:06:05.825 - 18:06:06.294  (469 ms, 6 events)
18:06:09.325 - 18:06:09.966  (641 ms, 6 events)
18:06:21.076 - 18:06:22.060  (984 ms, 5 events)
18:06:25.061 - 18:06:26.029  (968 ms, 13 events)
18:06:26.123 - 18:06:27.108  (985 ms, 15 events)
18:06:28.092 - 18:06:28.233  (141 ms, 16 events)
```

#### Case 4: M = 10, T = 50 ms (7 Bursts)

```text
18:01:53.822 - 18:01:53.869  (47 ms, 15 events)
18:01:53.869 - 18:01:53.885  (16 ms, 16 events)
18:01:58.916 - 18:01:58.963  (47 ms, 16 events)
18:03:55.408 - 18:03:55.423  (15 ms, 10 events)
18:03:55.923 - 18:03:55.939  (16 ms, 11 events)
18:06:26.108 - 18:06:26.139  (31 ms, 14 events)
18:06:28.217 - 18:06:28.248  (31 ms, 17 events)
```

#### Case 5: M = 10, T = 200 ms (15 Bursts)

```text
18:01:50.556 - 18:01:50.728  (172 ms, 12 events)
18:01:51.775 - 18:01:51.963  (188 ms, 14 events)
18:01:53.822 - 18:01:53.869  (47 ms, 15 events)
18:01:53.869 - 18:01:53.885  (16 ms, 16 events)
18:01:58.916 - 18:01:58.963  (47 ms, 16 events)
18:02:00.963 - 18:02:01.041  (78 ms, 16 events)
18:02:47.497 - 18:02:47.685  (188 ms, 10 events)
18:03:55.408 - 18:03:55.595  (187 ms, 16 events)
18:03:55.923 - 18:03:56.064  (141 ms, 12 events)
18:03:56.361 - 18:03:56.548  (187 ms, 13 events)
18:04:04.877 - 18:04:04.986  (109 ms, 12 events)
18:04:07.908 - 18:04:08.065  (157 ms, 12 events)
18:04:08.987 - 18:04:09.096  (109 ms, 11 events)
18:06:25.998 - 18:06:26.139  (141 ms, 24 events)
18:06:28.092 - 18:06:28.233  (141 ms, 16 events)
```

#### Case 6: M = 10, T = 1000 ms (22 Bursts)

```text
18:01:50.353 - 18:01:51.322  (969 ms, 19 events)
18:01:51.791 - 18:01:52.728  (937 ms, 15 events)
18:01:53.400 - 18:01:53.775  (375 ms, 16 events)
18:01:53.822 - 18:01:53.869  (47 ms, 15 events)
18:01:53.869 - 18:01:53.885  (16 ms, 16 events)
18:01:53.885 - 18:01:54.791  (906 ms, 16 events)
18:01:56.916 - 18:01:57.447  (531 ms, 21 events)
18:01:58.916 - 18:01:58.963  (47 ms, 16 events)
18:02:00.963 - 18:02:01.041  (78 ms, 16 events)
18:02:47.497 - 18:02:47.841  (344 ms, 15 events)
18:03:55.408 - 18:03:55.939  (531 ms, 16 events)
18:03:55.939 - 18:03:56.455  (516 ms, 15 events)
18:03:56.798 - 18:03:57.564  (766 ms, 14 events)
18:04:04.877 - 18:04:05.830  (953 ms, 16 events)
18:04:07.908 - 18:04:08.877  (969 ms, 19 events)
18:04:08.987 - 18:04:09.955  (968 ms, 16 events)
18:04:10.940 - 18:04:11.924  (984 ms, 12 events)
18:04:50.208 - 18:04:51.114  (906 ms, 16 events)
18:04:51.755 - 18:04:52.552  (797 ms, 12 events)
18:06:25.061 - 18:06:26.029  (968 ms, 13 events)
18:06:26.123 - 18:06:27.108  (985 ms, 15 events)
18:06:28.092 - 18:06:28.233  (141 ms, 16 events)
```

---

## 5. Discussion

### 5.1 Discussion of Distributed K-Way Log Stream Merger Results

The merger produced one sorted stream of all 2000 entries, running from
18:01:47.978 to 18:10:55.202. The merged timeline tells a clear story about the
job:

- **Start-up (first 20 entries):** every entry comes from **p4**
  (`MRAppMaster`). The application master is created, loads its configuration,
  and registers its event handlers before any other component logs anything.
  This matches p4's active period of 18:01:47–18:01:53.
- **Execution (≈ 18:01:53 – 18:06:28):** the streams interleave heavily as the
  job runs. The speculator and task attempts (p3), container allocation (p1),
  container launching (p6), and task heartbeats (p5) all log at once. Across
  the whole merged log, the source changes between consecutive entries 838
  times out of 1999 adjacent pairs. A simple concatenation of the files would
  therefore misrepresent the real ordering of events.
- **Failure (last 20 entries):** the tail contains only **p1** and **p2**,
  alternating in a steady loop about once per second. p1 cannot reach the
  ResourceManager (`msra-sa-41:8030`, _"ERROR IN CONTACTING RM"_), and p2 keeps
  failing to renew its HDFS lease on the NameNode (`msra-sa-41:9000`). The
  first WARN in the dataset appears at 18:05:27.570, and from then on the job
  is effectively stuck retrying. Of the 150 ERROR entries, 148 come from p1.
  Only the merged view makes it clear that both connection failures happen
  together and that the rest of the system has gone silent.

**Lazy loading in practice.** Because the readers and merges are generators,
at most one entry per file and one per merge node is in memory at any time
(O(k) = O(7) entries). Entries are written to disk as soon as they are
produced, so the same code would handle log files much larger than memory. The
trade-off is that a lazy stream can only be read forward once. Parts 2 and 3
therefore read the merged _file_ instead of reusing the stream, and they also
read it lazily, row by row.

**Ties.** Many entries share a millisecond timestamp (e.g. four p4 entries at
18:01:50,556). The `≤` comparison in `MERGE_TWO` makes the merge stable.
Entries from the same file keep their original order, and entries from
different files with the same timestamp are ordered by file index. Any order
among equal timestamps is valid, but a deterministic one makes the dependency
matrix reproducible.

### 5.2 Discussion of Using Strassen's Matrix Multiplication

**Dependency structure (A).** Thirty of the 42 possible off-diagonal
dependencies are present, so the component graph is dense, which is typical of
a tightly coupled MapReduce job.

- **p3** (task attempts, speculator, rack resolver) has the most outgoing
  dependencies: it is followed by every other component. This fits its role as
  the central coordinator of task state changes.
- **p4** (`MRAppMaster`) has only two outgoing edges (to p0 and p3) and is
  reached only from p0 and p3. It is active only during the first six seconds,
  so it takes part in very few transitions.
- **p6** (`ContainerLauncher`) is reached only from p3 and p5. Container
  launches are triggered by task-attempt events and task heartbeats, not by
  allocator (p1) or HDFS client (p2) activity.
- **p2** (HDFS client and lease renewer) depends only on p0, p1 and p5. Its
  activity is confined to the failure phase, where it alternates almost
  entirely with p1.

**Multi-hop dependencies (A<sup>2</sup>, A<sup>3</sup>).** Each entry of
_A<sup>m</sup>_ counts the number of dependency paths of length _m_, not just
whether a path exists.

- Every entry of _A<sup>2</sup>_ is positive: **every component can affect
  every other component (including itself) within two hops.**
- The positive diagonal of _A<sup>2</sup>_
  (e.g. _A<sup>2</sup>\[p0\]\[p0\] = 5_) shows short feedback cycles such as
  p0 → p1 → p0.
- In _A<sup>3</sup>_ the counts grow quickly (up to 19). The largest values are
  in rows p3, p5 and p6, which are the most connected components.
- The p4 row and column stay small in every power
  (_A<sup>3</sup>\[p4\]\[p4\] = 2_), so a fault starting in or flowing into the
  application master has the fewest propagation paths.
- From a fault-analysis point of view, p3 is the component whose failure would
  spread to the others through the most paths.

To get plain reachability (yes or no) instead of path counts, threshold the
matrices at _&gt; 0_.

**Strassen for arbitrary sizes.** Padding odd dimensions with a single zero row
and column at each level worked well. The 7 × 7 matrix goes through
8 → 4 → 2 → 1, and the padding is removed on the way back up. The results match
the naive product exactly, as expected, because Strassen's algorithm is exact
for integers.

For a matrix this small, Strassen is _not_ faster in practice. Each level does
18 block additions or subtractions and allocates new sub-matrices. In pure
Python that overhead outweighs saving one multiplication out of eight.
Strassen's _O(n<sup>2.807</sup>)_ advantage only shows up for much larger
matrices, typically hundreds of rows. Real implementations therefore switch to
the standard algorithm below a cutoff size; we used _n = 1_ to keep the
recursion easy to follow. Strassen also adds subtractions, so the intermediate
values can be negative even though _A_ is non-negative. This is harmless for
integers but causes rounding error with floating-point numbers.

### 5.3 Discussion of Divide-and-Conquer Burst Detection Results

**Effect of T and M.** The results behave as expected:

- **Larger T** gives each burst a wider window, so more groups of events
  qualify. For M = 5, the count rises from 18 to 22 to 40 bursts, and coverage
  from 8.9% to 22.5% of all events.
- **Larger M** keeps only the densest groups. For each T, M = 10 roughly halves
  the count (7, 15, 22).
- **Small T, large M** (case 4: 7 bursts) isolates the extreme spikes: 10–17
  events within 50 ms.
- **Large T, small M** (case 3: 40 bursts) also picks up moderate activity,
  such as groups of 5 events spread over almost a second during task execution
  between 18:02:49 and 18:03:46.

**Where the bursts are.** Every burst in every case falls between 18:01:50 and
18:06:28, the start-up and execution phases. The densest bursts line up with
clear job events:

- _18:01:53.822 – 18:01:53.885:_ the job moves to RUNNING and p3 (rack
  resolver, task and task-attempt handlers) sets up all map tasks at once. 43
  of the 44 events in this range come from p3.
- _18:01:58.9 / 18:02:01.0:_ containers are allocated and launched (p1, p6).
- _18:03:55 – 18:04:10:_ mostly container allocator (p1) and task listener (p5)
  activity, with task attempts (p3) and container launches (p6), as map tasks
  finish and new containers are scheduled.
- _18:06:25.998 – 18:06:26.139 (24 events)_ and _18:06:28.092 – 18:06:28.233
  (16 events)_: these contain the dataset's **only two FATAL entries** (p5,
  18:06:26.029 and 18:06:28.217, _"Task … exited:
  java.net.NoRouteToHostException"_). The tasks lose their route to the host,
  and many components log at once in response. These are the most interesting
  bursts for fault analysis.

The long failure phase after 18:06:28 produces **no bursts at all**, even with
T = 1000 ms and M = 5. The p1/p2 retry loop logs about 3–4 events per second at
regular intervals, which is never 5 events in under a second. Burst detection
therefore finds _sudden spikes_ in activity, not _ongoing_ failures. A system
stuck in a steady retry loop looks quiet to a density-based detector, so burst
detection should be combined with level-based checks (WARN/ERROR counts) to
catch both kinds of problem.

**Split bursts and non-overlap.** The combine step is essential:

- A base case holds at most _B = 20_ entries, so every burst with more than 20
  events must be a cross-boundary burst. Examples are the 24-event burst at
  18:06:25.998 and the 21-event burst at 18:01:56.916.
- Many smaller bursts also cross a partition boundary, since a median split
  rarely falls neatly between two dense regions.

Because the combine step uses only unclaimed events, all bursts in every list
are disjoint. The total number of events covered in a case never exceeds 2000,
and no event is counted twice.

**Limitations.** Requiring non-overlapping bursts has a side effect. A dense
region with more than about _2B_ events, or one that straddles several
partition boundaries, can be reported as several adjacent bursts instead of
one. For example, at T = 200 ms the bursts _18:01:53.822–.869_ (15 events) and
_18:01:53.869–.885_ (16 events) are listed separately. Together they span only
63 ms and would also qualify as a single 31-event burst. The algorithm counts
the non-overlapping bursts its partitioning finds, not maximal clusters, so the
exact count depends on _B_ and on where the median splits fall. A larger _B_
(or a final pass that joins adjacent bursts whose combined span is under _T_)
would report such regions as one burst. The trends across T and M, and where
the bursts occur, do not depend on this choice.

---

## 6. Conclusions

### 6.1 Key Lessons Learned

- **Divide-and-conquer is a general pattern.** The same divide / conquer /
  combine structure solved three different problems: merging (divide the set of
  streams), matrix multiplication (divide the matrix into quadrants), and burst
  detection (divide the timeline at the median). In each case the combine step
  held the problem-specific logic: a two-way merge, Strassen's seven-product
  formulas, and the cross-boundary burst check.
- **Lazy evaluation with generators.** Python generators made it easy to build
  a lazy merge tree with _O(k)_ memory, and to see how a recursion can be built
  once and then pulled through one element at a time.
- **Recurrences and complexity.** The three algorithms follow the recurrences
  _T(k) = 2T(k/2) + O(n)_ (merge, O(N log k)),
  _T(n) = 7T(n/2) + O(n<sup>2</sup>)_ (Strassen, O(n<sup>2.807</sup>)) and
  _T(n) = 2T(n/2) + O(n)_ (bursts, O(N log N)). Saving a single recursive call
  (7 instead of 8) changes the exponent of the running time.
- **Asymptotic vs. practical performance.** Strassen's better exponent only
  pays off for large matrices. For small inputs, constant factors and memory
  allocation dominate.
- **Graph analysis with matrix powers.** Powers of an adjacency matrix count
  multi-hop paths, which gives a compact way to measure how faults could spread
  between components.
- **Combine steps must respect the problem definition.** Returning bursts as
  index ranges instead of plain counts was the key to keeping split bursts
  non-overlapping.

### 6.2 Challenges Encountered

- **Defining cross-boundary bursts consistently.** The hardest design question
  was how to count bursts that span a partition boundary without
  double-counting events a base case had already claimed. The events within _T_
  of the boundary can span up to _2T_, so it was also not enough to test them
  all at once. The solution was to return disjoint index ranges from each
  recursive call, limit the combine step to unclaimed events, and use a
  two-pointer search for the largest valid crossing window.
- **Strassen for non-power-of-2 sizes.** We chose to pad odd dimensions by one
  at each recursion level rather than padding to the next power of 2 up front.
  This required carefully removing the padding after each recursive call.
- **Keeping the merge truly lazy.** Every stage had to be a generator, and the
  output had to be written as it was produced. Collecting results into a list
  anywhere would have quietly broken the lazy-loading requirement. A related
  Python pitfall came up while building the toy example: a generator expression
  built in a loop captured the loop variable late, so every toy stream reported
  the same source. The fix was to build each toy stream's entries immediately.
- **Timestamp parsing and ties.** The `Time` field uses a comma before the
  milliseconds (`18:01:51,775`), and the CSV content fields themselves contain
  commas and quotes. A real CSV parser was needed, and timestamps were
  converted to integer milliseconds. Many entries share a timestamp, so a
  stable tie-breaking rule was needed to keep the merged order, and therefore
  the dependency matrix, deterministic.
- **Choosing experiment parameters.** T, M and B had to be picked so that the
  six cases showed meaningful differences. T spans three orders of magnitude
  (50 ms to 1 s), and M = 5 vs. 10 separates moderate activity from extreme
  spikes. B = 20 kept base cases larger than M so local bursts could form.
