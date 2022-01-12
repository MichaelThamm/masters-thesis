from Temp1 import*

from numba import jit, njit, vectorize
from timeit import default_timer as timer
import contextlib
import numpy as np
import random


@contextlib.contextmanager
def timing():
    s = timer()
    yield
    e = timer()
    print('execution time: {:.2f}s'.format(e - s))


# @njit(cache=True)
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    acc += temp1Add(np.arange(1000))
    return 4.0 * acc / nsamples


with timing():
    monte_carlo_pi(10000)