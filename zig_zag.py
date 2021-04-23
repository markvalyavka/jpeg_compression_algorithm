import numpy as np


def zig_zag(arr):
    """
    reorder DCT coefficients in zig-zag order by calling zigzag function
    returns one-dimensional array
    """
    k, i, j = 0, 0, 0
    length = len(arr[0])
    height = len(arr)

    result = np.zeros((height * length))
    while i < length and j < height:
        result[k] = arr[j, i]
        if (i + j) % 2 == 0:  # going up
            if j > 0 and i < length - 1:
                i += 1
                j -= 1

            elif j >= 0 and i == height - 1:
                j += 1

            elif j > 0 and i == height - 1:
                j += 1

            elif j == 0 and i <= height - 1:
                i += 1
        else:  # going down
            if i > 0 and j < height - 1:
                i -= 1
                j += 1

            elif i > 0 and j == length - 1:
                i += 1

            elif i >= 0 and j == length - 1:
                i += 1

            elif i == 0 and j <= length - 1:
                j += 1

        k += 1
    return result
