import multiprocessing
import numpy as np

from os import cpu_count


def worker(A_rows, x, result, start_idx):
    """Worker function to compute partial matrix-vector product.

    Arguments
    ---------
    A_rows : numpy.ndarray
        Subset of rows from matrix A.
    x : numpy.ndarray
        The input vector.
    result : multiprocessing.Array
        Shared array to store results.
    start_idx : int
        Starting index in result array to write to.
    """
    partial_result = A_rows @ x
    for i, val in enumerate(partial_result):
        result[start_idx + i] = val


def parallel_matvec(A: np.ndarray, x: np.ndarray, nproc: int):
    """Compute the matrix-vector product A * x in parallel.

    Arguments
    ---------
    A : numpy.ndarray
        The input matrix.
    x : numpy.ndarray
        The input vector.
    nproc : int
        The number of parallel processes to use.

    Returns
    -------
    numpy.ndarray
        A vector holding the result of the operation `A * x`.

    Raises
    ------
    ValueError
        If the number of processes is not strictly positive.
    """
    if nproc <= 0:
        raise ValueError("Number of processes must be strictly positive")

    m, n = A.shape
    rows_per_proc = m // nproc

    # Initialize a shared array to store results
    result = multiprocessing.Array('d', m)

    processes = []
    for i in range(nproc):
        start_row = i * rows_per_proc
        if i == nproc - 1:
            # Last process handles remaining rows
            end_row = m
        else:
            end_row = (i + 1) * rows_per_proc

        A_rows = A[start_row:end_row, :]
        proc = multiprocessing.Process(target=worker, args=(A_rows, x, result, start_row))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    return np.array(result[:])


if __name__ == "__main__":
    A = np.random.randn(100, 50)
    x = np.random.rand(50)
    y_numpy = A @ x
    nproc = cpu_count()
    if nproc:
        y_nonnumpy = parallel_matvec(A, x, 2 * nproc)
    else:
        y_nonnumpy = parallel_matvec(A, x, 1)
    print(np.linalg.norm(y_nonnumpy - y_numpy))
