import scipy
import numpy as np


MOORE = np.ones((3, 3), dtype = np.uint8)
MOORE[1, 1] = 0


def next_state(prev):
    neighbours = scipy.signal.convolve2d(prev, MOORE, mode = "same", boundary = "fill", fillvalue = 0)
    state = np.zeros_like(prev)
    state[np.where((neighbours == 2) & (prev == 1))] = 1
    state[np.where((neighbours == 3) & (prev == 1))] = 1
    state[np.where((neighbours == 3) & (prev == 0))] = 1
    return state

