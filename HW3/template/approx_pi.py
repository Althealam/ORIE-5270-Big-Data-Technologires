import multiprocessing

from math import ceil
from os import cpu_count
from random import uniform

def trial(m):
    """Approximate pi using a Monte-Carlo method.

    This function generates `m` random samples on the unit square and counts
    how many of said samples ended up inside the unit circle.

    Arguments
    ---------
    m : int
        The number of random samples to generate.

    Returns
    -------
    int
        The number of samples that fell inside the unit circle.

    Raises
    ------
    ValueError
        If the number m is not strictly positive.
    """
    if m <= 0:
        raise ValueError("Number of samples must be strictly positive")

    count = 0
    for _ in range(m):
        x = uniform(-1, 1)
        y = uniform(-1, 1)
        if x*x + y*y <= 1:
            count += 1
    return count


def approx_pi(N):
    """Approximate pi using multiple processes and N samples total.

    The number of processes generated should be equal to the number of CPU
    cores available to the user.

    Arguments
    ---------
    N : int
        The total number of samples to use.

    Returns
    -------
    float
        The computed estimate for pi.

    Raises
    ------
    ValueError
        If the number of samples `N` is not positive.
    """
    if N <= 0:
        raise ValueError("Number of samples must be positive")

    nproc = cpu_count() or 1
    samples_per_proc = N // nproc

    # Use Pool.map to run trials in parallel
    with multiprocessing.Pool(processes=nproc) as pool:
        counts = pool.map(trial, [samples_per_proc] * nproc)

    total_inside = sum(counts)
    pi_estimate = 4 * total_inside / N
    return pi_estimate


def approx_pi_bonus(*number_of_samples):
    """Approximate pi using multiple processes.

    The number of processes generated must be equal to the number of arguments
    passed to this function. Each argument must be a positive integer that
    specifies how many samples the corresponding process should use.

    Arguments
    ---------
    number_of_samples : array-like
        A variable-length array of sample counts.

    Returns
    -------
    float
        The computed estimate for pi.

    Raises
    ------
    ValueError
        If any of the arguments is not positive or if no arguments have been
        provided.
    """
    if len(number_of_samples) == 0:
        raise ValueError("No arguments provided")

    for n in number_of_samples:
        if n <= 0:
            raise ValueError("All sample counts must be positive")

    k = len(number_of_samples)

    # Use Pool.map to run trials in parallel
    with multiprocessing.Pool(processes=k) as pool:
        counts = pool.map(trial, number_of_samples)

    total_inside = sum(counts)
    total_samples = sum(number_of_samples)
    pi_estimate = 4 * total_inside / total_samples
    return pi_estimate


if __name__ == "__main__":
    pi_est_1 = approx_pi(1000)
    pi_est_2 = approx_pi_bonus(250, 250, 100, 300, 100)
    print(pi_est_1, pi_est_2)
