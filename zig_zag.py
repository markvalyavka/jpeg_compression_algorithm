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


# Algorithm from @getsanjeev github
#
def inverse_zigzag(input, vmax, hmax) :
    """
    reorder DCT coefficients in inverse zig-zag order
    returns: two-dimensional matrix
    """

    h, v, vmin, hmin = 0, 0, 0, 0
    result = np.zeros((vmax, hmax))

    i = 0
    while (v < vmax) and (h < hmax):

        if ((h + v) % 2) == 0:  # going up

            if v == vmin:
                result[v, h] = input[i]
                if h == hmax:
                    v += 1
                else:
                    h += 1

                i += 1

            elif (h == hmax - 1) and (v < vmax):
                result[v, h] = input[i]
                v += 1
                i += 1

            elif (v > vmin) and (h < hmax - 1):
                # print(3)
                result[v, h] = input[i]
                v -= 1
                h += 1
                i += 1
        else:

            if (v == vmax - 1) and (h <= hmax - 1):

                result[v, h] = input[i]
                h += 1
                i += 1

            elif h == hmin:

                result[v, h] = input[i]
                if v == vmax - 1:
                    h += 1
                else:
                    v += 1
                i += 1

            elif (v < vmax - 1) and (h > hmin):
                result[v, h] = input[i]
                v += 1
                h -= 1
                i += 1

        if (v == vmax - 1) and (h == hmax - 1):
            result[v, h] = input[i]
            break

    return result
