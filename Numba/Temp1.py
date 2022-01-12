import numpy as np
from numba import njit

# @njit
def temp1Add(array):
    print(array.shape)
    val = 0
    for ii in array:
        val += ii
    return val