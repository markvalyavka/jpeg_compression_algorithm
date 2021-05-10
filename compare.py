import cv2
import numpy as np
import math


def perc_storage(rank, n_rows, n_cols):
    original_space = n_rows*n_cols
    compressed_space = n_rows*rank + rank + n_cols*rank
    return compressed_space / original_space * 100


def perc_energy(S, r):
    return (np.trace(S[:r]) / np.trace(S)) * 100


