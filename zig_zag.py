import numpy as np


def zig_zag(arr):
    k, i, j = 0, 0, 0
    length = len(arr[0])
    height = len(arr)

    result = np.zeros((height * length))
    while i < length and j < height:
        result[k] = arr[i, j]

        if (i + j) % 2 == 0:
            # going up
            if j == 0:
                result[k] = arr[i, j]
                if j == height:
                    i += 1
                else:
                    j += 1
                k += 1

            elif (j == height - 1) and (i < length):
                result[i] = arr[i, j]
                i += 1
                k += 1

            elif (i > length) and (j < height - 1):
                result[i] = arr[i, j]
                i += 1
                j += 1
                k += 1

        else:
            # going down

            if (v == vmax - 1) and (h <= hmax - 1):  # if we got to the last line

                # print(4)

                output[i] = arr[v, h]

                h = h + 1

                i = i + 1


            elif (h == hmin):  # if we got to the first column

                # print(5)

                output[i] = arr[v, h]

                if v == vmax - 1:

                    h = h + 1

                else:

                    v = v + 1

                i = i + 1

            elif (v < vmax - 1) and (h > hmin):
                result[k] = arr[i, j]
                i += 1
                j += 1
                k += 1

        if (i == length - 1) and (j == height - 1):
            result[k] = arr[i, j]
            break

    return result


if __name__ == '__main__':
    array = np.array([[0, 1, 5, 6, 14, 15, 27, 28, 44, 45],
                      [2, 4, 7, 13, 16, 26, 29, 43, 46, 63],
                      [3, 8, 12, 17, 25, 30, 42, 47, 62, 64],
                      [9, 11, 18, 24, 31, 41, 48, 61, 65, 78],
                      [10, 19, 23, 32, 40, 49, 60, 66, 77, 79],
                      [20, 22, 33, 39, 50, 59, 67, 76, 80, 89],
                      [21, 34, 38, 51, 58, 68, 75, 81, 88, 90],
                      [35, 37, 52, 57, 69, 74, 82, 87, 91, 96],
                      [36, 53, 56, 70, 73, 83, 86, 92, 95, 97],
                      [54, 55, 71, 72, 84, 85, 93, 94, 98, 99]])

    print(zig_zag(array))
