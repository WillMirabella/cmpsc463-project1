"""Part 2: Fault Dependency Analysis using Strassen's Matrix Multiplication."""
import csv


def build_dependency_matrix(merged_path, k):
    """Build the k x k binary matrix A where A[x][y] = 1 if an event from component y
    immediately follows an event from component x in the merged timeline (x != y)."""
    A = [[0] * k for _ in range(k)]
    prev = None
    with open(merged_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            cur = int(row[0])  # Source column
            # self-transitions are skipped since a component trivially depends on itself
            if prev is not None and prev != cur:
                A[prev][cur] = 1
            prev = cur
    return A


def add(X, Y):
    return [[X[i][j] + Y[i][j] for j in range(len(X))] for i in range(len(X))]


def sub(X, Y):
    return [[X[i][j] - Y[i][j] for j in range(len(X))] for i in range(len(X))]


def naive_multiply(X, Y):
    """Standard O(n^3) multiplication, used to verify Strassen's results."""
    n = len(X)
    return [[sum(X[i][k] * Y[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def strassen(X, Y):
    """Multiply two n x n matrices with Strassen's algorithm for any n."""
    n = len(X)
    if n == 1:  # base case
        return [[X[0][0] * Y[0][0]]]

    # odd dimension: pad with a zero row/column, recurse, then strip the padding
    if n % 2 == 1:
        Xp = [row + [0] for row in X] + [[0] * (n + 1)]
        Yp = [row + [0] for row in Y] + [[0] * (n + 1)]
        C = strassen(Xp, Yp)
        return [row[:n] for row in C[:n]]

    # dividing each matrix into four n/2 x n/2 quadrants
    h = n // 2
    A11 = [row[:h] for row in X[:h]]
    A12 = [row[h:] for row in X[:h]]
    A21 = [row[:h] for row in X[h:]]
    A22 = [row[h:] for row in X[h:]]
    B11 = [row[:h] for row in Y[:h]]
    B12 = [row[h:] for row in Y[:h]]
    B21 = [row[:h] for row in Y[h:]]
    B22 = [row[h:] for row in Y[h:]]

    # 7 recursive products instead of 8
    M1 = strassen(add(A11, A22), add(B11, B22))
    M2 = strassen(add(A21, A22), B11)
    M3 = strassen(A11, sub(B12, B22))
    M4 = strassen(A22, sub(B21, B11))
    M5 = strassen(add(A11, A12), B22)
    M6 = strassen(sub(A21, A11), add(B11, B12))
    M7 = strassen(sub(A12, A22), add(B21, B22))

    # combining the products into the result quadrants
    C11 = add(sub(add(M1, M4), M5), M7)
    C12 = add(M3, M5)
    C21 = add(M2, M4)
    C22 = add(sub(add(M1, M3), M2), M6)

    # stitch quadrants back into one n x n matrix
    top = [C11[i] + C12[i] for i in range(h)]
    bottom = [C21[i] + C22[i] for i in range(h)]
    return top + bottom
