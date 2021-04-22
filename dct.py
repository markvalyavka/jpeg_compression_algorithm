import numpy as np
from quantization import get_quantization_matrix
from math import cos, sqrt, pi


def calculate_dct_matrix_entry(u, v, n):
    if u == 0:
        return sqrt(1 / n)
    else:
        return sqrt(2 / n) * cos(((2 * v + 1) * pi * u) / (2 * n))


def get_dct_matrix(n):
    """
    Get dct n-dimensional coefficient matrix C

    :param n: dimension of dct coefficient matrix
    :return: dct coefficient matrix
    """
    C = np.zeros((n, n))

    for u in range(n):
        for v in range(n):
            C[u, v] = round(calculate_dct_matrix_entry(u, v, n), 2)
    return C


def forward_dct(blocks):
    DCT_T = get_dct_matrix(8)
    Q = get_quantization_matrix()

    for block_row in range(len(blocks)):
        for block_col in range(len(blocks[0])):
            # Normalize values around 0 by subtracting 128 from each entry
            # Needed because DCT works with values in range [-128, 127]
            blocks[block_row][block_col] -= 128
            # Perform DCT by using matrix multiplication
            # D = T M T'
            blocks[block_row][block_col] = np.linalg.multi_dot([DCT_T, blocks[block_row][block_col], DCT_T.transpose()])
            # Quantize using quantization matrix Q
            blocks[block_row][block_col] = np.round(np.divide(blocks[block_row][block_col], Q))


def backwards_dct(blocks):
    DCT_T = get_dct_matrix(8)
    Q = get_quantization_matrix()

    for block_row in range(len(blocks)):
        for block_col in range(len(blocks[0])):
            # De-quantization step (Where error happens)
            blocks[block_row][block_col] = np.multiply(blocks[block_row][block_col], Q)
            # Invert DCT
            blocks[block_row][block_col] = np.linalg.multi_dot([DCT_T.transpose(), blocks[block_row][block_col], DCT_T])
            # Add 128 back
            blocks[block_row][block_col] += 128

