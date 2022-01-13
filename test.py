import concurrent.futures
from timeit import default_timer as timer
import contextlib
from itertools import chain, combinations
from scipy.linalg import lu_factor, lu_solve
import numpy as np
import cmath


@contextlib.contextmanager
def timing():
    s = timer()
    yield
    e = timer()
    print('execution time: {:.5f}s'.format(e - s))


def f(x):
    res = 0
    for i in range(x):
        res = i ** i

    return res


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(len(ss) + 1)))


if __name__ == '__main__':
    bool_ = True
    with timing():
        if bool_:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [executor.submit(f, i) for i in range(10)]

                for f in concurrent.futures.as_completed(results):
                    print(f.result())
        else:
            print(list(map(f, [i for i in range(10)])))
