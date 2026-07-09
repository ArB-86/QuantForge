import os


def get_num_threads(default=8):

    omp = os.environ.get(
        "OMP_NUM_THREADS"
    )

    if omp:

        return int(omp)

    return default
